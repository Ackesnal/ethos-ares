from copy import copy
from pathlib import Path
from queue import Queue

import torch as th

from ..constants import SpecialToken as ST
from ..utils import load_model_checkpoint, setup_torch
from ..vocabulary import Vocabulary
from .constants import Reason, Task
from .utils import create_loader, get_dataset_cls, get_next_token, get_token_time


def spawn_inference_worker(
    job_queue: Queue,
    model_fp: str | Path,
    task: Task,
    dataset_kwargs: dict,
    progress_queue: Queue,
    temperature: float = 1.0,
    rep_num: int = 1,
    device: str = "cuda",
    no_compile: bool = False,
    save_generated_tokens: bool = False,
):
    if "cuda" in device:
        th.cuda.set_device(device)
        th.set_float32_matmul_precision("high")
    autocast_context = setup_torch(device, dtype="bfloat16" if "cuda" in device else "float32")

    model, _ = load_model_checkpoint(model_fp, map_location=device)
    model.to(device)
    model = th.compile(model, disable=no_compile)

    dataset_cls = get_dataset_cls(task)
    # TODO: Validate if the vocab in the dataset is the same as in the model
    dataset = dataset_cls(**dataset_kwargs)

    max_timeline_size = dataset_kwargs["n_positions"]
    ctx_size = dataset.context_size
    vocab: Vocabulary = dataset.vocab

    data_loader = create_loader(job_queue, dataset)

    stop_stokens = dataset.stop_stokens
    stop_tokens = th.tensor(vocab.encode(stop_stokens), dtype=th.long)

    time_limit = th.tensor(dataset.time_limit.total_seconds() * 1e6)

    for timeline_pack, ground_truth in data_loader:
        # Unpack timeline data and times --------------------------------
        # timeline_pack is always (timeline_data, timeline_times) where
        #   decoder-only:      timeline_data = 1-D token tensor
        #   encoder-decoder:   timeline_data = (pt_ctx, timeline_tokens)
        timeline_data, timeline_times = timeline_pack

        ctx = None
        if isinstance(timeline_data, tuple):
            ctx, timeline_data = tuple(t.to(device, non_blocking=True) for t in timeline_data)
            ctx = ctx.repeat(rep_num, 1)
        else:
            timeline_data = timeline_data.to(device, non_blocking=True)
        timeline_times = timeline_times.to(device, non_blocking=True)

        timeline = timeline_data.repeat(rep_num, 1)
        timeline_times = timeline_times.repeat(rep_num, 1)

        gen_token_num, offset = 0, 0
        gen_times = th.zeros(rep_num, dtype=th.float64)
        generated_tokens = [] if save_generated_tokens else None
        while timeline.size(0):
            if task == Task.SOFA_PREDICTION and gen_token_num == 1:
                # append a sofa token to the timeline and continue generating
                next_token = th.tensor(
                    [vocab.encode([ST.SOFA])], device=timeline.device, dtype=th.long
                )
                next_token = next_token.repeat(timeline.size(0), 1)
            else:
                with autocast_context:
                    next_token, probs = get_next_token(
                        model, timeline, ctx=ctx, times=timeline_times,
                        return_probs=True, temperature=temperature,
                    )

            if generated_tokens is not None:
                generated_tokens.append(next_token)

            if not offset and timeline.size(1) == max_timeline_size:
                offset = 1

            # Estimate the time for the newly generated token and extend
            # timeline_times in lockstep with timeline.
            new_token = next_token.cpu().view(-1)
            new_token_time_delta = get_token_time(new_token, vocab)  # μs
            # Use the last known time + estimated delta as the new token's time
            last_time = timeline_times[:, -1].float()  # (rep_num,)
            next_time = (last_time + new_token_time_delta.to(last_time.device)).long()
            next_time_col = next_time.unsqueeze(1)  # (rep_num, 1)

            if ctx is not None:
                new_timeline = (timeline[:, offset:], next_token)
                new_times = (timeline_times[:, offset:], next_time_col)
            else:
                new_timeline = (
                    timeline[:, :ctx_size],
                    timeline[:, ctx_size + offset :],
                    next_token,
                )
                new_times = (
                    timeline_times[:, :ctx_size],
                    timeline_times[:, ctx_size + offset :],
                    next_time_col,
                )
            timeline = th.cat(new_timeline, dim=1)
            timeline_times = th.cat(new_times, dim=1)

            gen_token_num += 1

            gen_times += new_token_time_delta

            completed_this_iter = th.isin(new_token, stop_tokens) | (gen_times > time_limit)

            if (task == Task.DRG_PREDICTION) or (
                task == Task.SOFA_PREDICTION and gen_token_num == 3
            ):
                completed_this_iter[:] = True

            if not completed_this_iter.any():
                continue

            for i in th.nonzero(completed_this_iter):
                stop_reason = Reason.GOT_TOKEN
                actual_token = next_token[i].item()

                token_time = gen_times[i]
                if token_time > time_limit:
                    stop_reason = Reason.TIME_LIMIT

                if th.isinf(token_time):
                    actual_stoken = str(actual_token)
                    stop_reason = Reason.KEY_ERROR
                    token_time = None
                else:
                    actual_stoken = vocab.decode(actual_token)
                    token_time = round(token_time.item())

                gt = copy(ground_truth)

                results = {
                    "expected": gt.pop("expected"),
                    "actual": actual_stoken,
                    "stop_reason": stop_reason,
                    "actual_prob": probs[i, actual_token].item(),
                    **dict(zip(stop_stokens, probs[i, stop_tokens].tolist())),
                    "true_token_time": gt.pop("true_token_time"),
                    "token_time": token_time,
                    "true_token_dist": gt.pop("true_token_dist"),
                    "token_dist": gen_token_num,
                    **gt,
                }

                if generated_tokens is not None:
                    results["generated_tokens"] = [tokens[i].item() for tokens in generated_tokens]

                progress_queue.put(results)

            if completed_this_iter.all():
                break

            not_completed_mask = ~completed_this_iter
            timeline = timeline[not_completed_mask, :]
            timeline_times = timeline_times[not_completed_mask, :]
            gen_times = gen_times[not_completed_mask]
            if generated_tokens is not None:
                generated_tokens = [tokens[not_completed_mask, :] for tokens in generated_tokens]
