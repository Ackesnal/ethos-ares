import math
import os
import time
from pathlib import Path

import hydra
import torch as th
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, GPT2Config

from ..datasets import TimelineDataset
from ..model import GPT2LMNoBiasModel
from ..utils import load_model_checkpoint, setup_torch
from .metrics import estimate_loss
from .utils import ModelType, configure_optimizers, estimate_mfu, get_lr, make_infinite_loader


def _convert_to_moe_state_dict(state_dict, num_experts):
    """Convert a non-MoE model state dict to MoE format.

    Copies backbone FFN weights (``.mlp.``) into every expert
    (``.moe.experts.{i}.``).  Router weights are left out so they keep
    their random initialisation in the target model.

    Works for both plain GPT2LMNoBiasModel state dicts *and*
    EncoderDecoderModel state dicts (where decoder keys are prefixed
    with ``decoder.``).
    """
    moe_state_dict = {}
    for key, value in state_dict.items():
        if ".mlp." in key:
            for expert_idx in range(num_experts):
                new_key = key.replace(".mlp.", f".moe.experts.{expert_idx}.")
                moe_state_dict[new_key] = value.clone()
        else:
            moe_state_dict[key] = value
    return moe_state_dict


def _compute_moe_batch_params(cfg, num_experts, device, master_process):
    """Compute adjusted batch_size and gradient_accumulation_steps for MoE stage 2.

    The MoE model has *num_experts* copies of each FFN layer, which can
    easily cause an out-of-memory error if the batch size is kept the
    same as stage 1.  We estimate the memory scale factor from the
    parameter-count ratio and reduce the batch size accordingly, while
    increasing gradient accumulation so the effective batch stays the
    same.

    Returns (new_batch_size, new_gradient_accumulation_steps).
    """
    # A rough but practical heuristic: model memory scales linearly
    # with the number of expert parameters.  In the backbone each layer
    # has 1 FFN; in the MoE model it has *num_experts* FFNs plus a
    # small router.  Non-FFN params (attention, embeddings, LN) stay
    # the same.  Typical GPT-2 blocks have ~2/3 FFN params, so:
    #   mem_ratio ≈ (1/3 + 2/3 * num_experts) / 1 = 1/3 + 2/3 * E
    # We add a small safety margin (1.15).
    ffn_fraction = 2.0 / 3.0
    non_ffn_fraction = 1.0 - ffn_fraction
    mem_ratio = non_ffn_fraction + ffn_fraction * num_experts
    safety_margin = 1.1
    scale_factor = mem_ratio * safety_margin

    original_batch_size = cfg.batch_size
    original_grad_accum = cfg.gradient_accumulation_steps
    effective_batch = original_batch_size * original_grad_accum

    # Scale batch size down (at least 1) to the exponential of 2
    new_batch_size = max(1, int(original_batch_size / scale_factor))
    new_batch_size = 2 ** math.ceil(math.log2(new_batch_size))
    # Increase gradient accumulation to compensate
    new_grad_accum = max(1, round(effective_batch / new_batch_size))

    if master_process:
        logger.info(
            f"Auto-scaling batch params for MoE Stage 2 (mem_ratio≈{mem_ratio:.2f}): "
            f"batch_size {original_batch_size}→{new_batch_size}, "
            f"gradient_accumulation_steps {original_grad_accum}→{new_grad_accum}"
        )

    return new_batch_size, new_grad_accum


def _build_moe_model(cfg, original_num_experts, vocab_size, model_type, train_dataset):
    """Build a fresh MoE model for Stage 2 (supports both decoder-only and encoder-decoder)."""
    moe_gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=cfg.n_positions,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=None,
        activation_function=cfg.activation,
        resid_pdrop=cfg.dropout,
        embd_pdrop=cfg.dropout,
        attn_pdrop=cfg.dropout,
        bias=False,
    )
    moe_gpt2_config.ffn_type = cfg.ffn_type
    moe_gpt2_config.num_experts_total = original_num_experts
    moe_gpt2_config.num_experts_activated = cfg.num_experts_activated

    if model_type == ModelType.ENC_DECODER:
        encoder_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=cfg.n_embd,
            num_hidden_layers=1,
            num_attention_heads=cfg.n_head,
            intermediate_size=cfg.n_embd * 4,
            hidden_act=cfg.activation,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
            max_position_embeddings=train_dataset.dataset.context_size,
            max_length=train_dataset.dataset.context_size,
            is_encoder_decoder=True,
            use_bfloat16=True,
        )
        enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, moe_gpt2_config
        )
        moe_model = EncoderDecoderModel(config=enc_dec_config)
    else:
        moe_model = GPT2LMNoBiasModel(moe_gpt2_config)

    return moe_model


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def main(cfg: DictConfig):
    """This training script can be run both on a single gpu in debug mode, and also in a larger
    training run with distributed data parallel (ddp).

    To run on a single GPU, example:
    $ ethos_train [args...]

    To run with DDP on 4 gpus on 1 node, example:

    $ torchrun --standalone --nproc_per_node=4 ethos_train [args...]

    To run with DDP on 4 gpus across 2 nodes, example:

    - Run on the first (master) node with example IP 123.456.123.456:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456
     --master_port=1234 ethos_train [args...]

    - Run on the worker node:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456
     --master_port=1234 ethos_train [args...]

    (If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
    """
    model_type = ModelType(cfg.model_type)

    device = cfg.device
    out_dir = Path(cfg.out_dir)
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=cfg.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        th.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert cfg.gradient_accumulation_steps % ddp_world_size == 0
        cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0

    tokens_per_iter = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.n_positions

    if master_process:
        logger.info(f"Tokens per iteration per worker: {tokens_per_iter:,}")
        out_dir.mkdir(parents=True, exist_ok=True)
    ctx = setup_torch(device, cfg.dtype, 42 + seed_offset)

    # --- MoE training-mode detection -----------------------------------------
    original_num_experts = cfg.num_experts_total
    if original_num_experts < 1:
        if master_process:
            logger.info(f"num_experts_total={original_num_experts} < 1, clamping to 1")
        original_num_experts = 1
        cfg.num_experts_total = 1

    two_stage_moe = original_num_experts > 1
    # When resuming we will detect which stage to enter below; for fresh
    # runs we always start with a plain backbone in Stage 1.
    resume_into_stage2 = False
    if two_stage_moe and not cfg.resume:
        cfg.num_experts_total = 1  # Stage 1 builds a plain (non-MoE) backbone
        if master_process:
            logger.info(
                f"MoE two-stage training enabled ({original_num_experts} experts). "
                "Stage 1: training backbone without MoE."
            )

    train_dataset = TimelineDataset(
        cfg.data_fp,
        n_positions=cfg.n_positions,
        is_encoder_decoder=model_type == ModelType.ENC_DECODER,
    )
    vocab = train_dataset.vocab

    vocab_size = math.ceil(len(vocab) / 64) * 64

    train_dataset, val_dataset = train_dataset.train_test_split(cfg.val_size)
    _pin_memory = "cuda" in device
    train_dataloader, val_dataloader = (
        DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=not ddp,
            sampler=DistributedSampler(dataset) if ddp else None,
            num_workers=8,
            pin_memory=_pin_memory,
            persistent_workers=True,
        )
        for dataset in [train_dataset, val_dataset]
    )
    train_dataloader = make_infinite_loader(train_dataloader)

    eval_iters = len(val_dataset) // (cfg.batch_size * cfg.n_positions) + 1
    if master_process:
        logger.info(
            "Train dataset size: {:,}, Val dataset size: {:,} (eval_iters={})".format(
                len(train_dataset), len(val_dataset), eval_iters
            )
        )

    def get_batch() -> tuple[th.Tensor | tuple, th.Tensor]:
        x, y = next(train_dataloader)
        y = y.to(device, non_blocking=True)
        if isinstance(x, list):
            return (x[0].to(device, non_blocking=True), x[1].to(device, non_blocking=True)), y
        return x.to(device, non_blocking=True), y

    iter_num, best_val_loss, best_metric_score, optimizer_state, wandb_path = 0, 1e9, 0, None, None
    if cfg.resume:
        model_fp = out_dir / "recent_model.pt"
        logger.info(f"Resuming from the most recent model: {model_fp}")
        raw_model, checkpoint = load_model_checkpoint(model_fp, map_location=device)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        best_metric_score = checkpoint["best_metric_score"]
        optimizer_state = checkpoint["optimizer"]
        wandb_path = checkpoint["wandb_path"]

        # Detect which stage the checkpoint belongs to
        saved_stage = checkpoint.get("training_stage", 1)
        saved_num_experts = checkpoint.get("original_num_experts", 1)
        if two_stage_moe:
            if saved_stage == 2:
                # Resume directly into Stage 2
                resume_into_stage2 = True
                if master_process:
                    logger.info(
                        f"Resuming into Stage 2 (MoE, {saved_num_experts} experts) "
                        f"at iter {iter_num}."
                    )
            else:
                # Resume into Stage 1 — ensure backbone (non-MoE) config
                cfg.num_experts_total = 1
                if master_process:
                    logger.info(
                        f"Resuming into Stage 1 (backbone) at iter {iter_num}."
                    )
    else:
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=cfg.n_positions,
            n_embd=cfg.n_embd,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_inner=None,
            activation_function=cfg.activation,
            resid_pdrop=cfg.dropout,
            embd_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout,
            bias=False,
        )
        # Extra attributes for FFN type and MoE
        config.ffn_type = cfg.ffn_type
        config.num_experts_total = cfg.num_experts_total
        config.num_experts_activated = cfg.num_experts_activated
        if model_type == ModelType.ENC_DECODER:
            encoder_config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=cfg.n_embd,
                num_hidden_layers=1,
                num_attention_heads=cfg.n_head,
                intermediate_size=cfg.n_embd * 4,
                hidden_act=cfg.activation,
                hidden_dropout_prob=cfg.dropout,
                attention_probs_dropout_prob=cfg.dropout,
                max_position_embeddings=train_dataset.dataset.context_size,
                max_length=train_dataset.dataset.context_size,
                is_encoder_decoder=True,
                use_bfloat16=True,
            )
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, config)
            raw_model = EncoderDecoderModel(config=config)
        else:
            raw_model = GPT2LMNoBiasModel(config)

        if master_process:
            logger.info(f"Initializing a new model from scratch: {config}")

    use_moe = getattr(raw_model, "use_moe", False)
    num_params_total = raw_model.num_parameters()
    if hasattr(raw_model, "num_active_parameters"):
        num_params_active = raw_model.num_active_parameters()
    else:
        num_params_active = num_params_total
    if master_process:
        logger.info(
            f"Model parameters — total: {num_params_total / 1e6:.2f}M, "
            f"active (per token): {num_params_active / 1e6:.2f}M"
        )

    # When resuming directly into Stage 2, we defer model placement,
    # compilation, and optimizer creation to the Stage 2 block.
    if not resume_into_stage2:
        raw_model.to(device)
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = th.amp.GradScaler(enabled=(cfg.dtype == "float16"))
        # optimizer
        optimizer = configure_optimizers(
            raw_model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        num_params = num_params_active
        if master_process:
            logger.info(("Not c" if cfg.no_compile else "C") + "ompiling the model...")
        model = th.compile(raw_model, disable=cfg.no_compile)

        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

    # logging
    online_logger, wandb_run = None, None
    if cfg.wandb_log and master_process:
        import wandb

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        dataset_name = Path(cfg.data_fp).parts[-2]
        cfg_dict.update(
            {
                "dataset": dataset_name,
                "vocab_size": len(vocab),
                "vocab_size_train": vocab_size,
                "model_num_params_active": num_params_active,
                "model_num_params_total": num_params_total,
                "model_num_params_total_with_emb": raw_model.num_parameters(exclude_embeddings=False),
            }
        )
        run_id = wandb_path.split("/")[-1] if wandb_path is not None else None
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg_dict,
            tags=[dataset_name],
            resume_from=f"{run_id}?_step={iter_num}" if run_id is not None else None,
        )
        online_logger = wandb

    # ---- training stage helper (closure over data loaders, cfg, ctx, …) ------
    def run_training_stage(
        model,
        raw_model,
        optimizer,
        scaler,
        use_moe,
        num_params,
        start_iter,
        max_iters_stage,
        best_val_loss,
        best_metric_score,
        lr_scale=1.0,
        stage_label="",
        training_stage=1,
        stage_batch_size=None,
        stage_grad_accum=None,
    ):
        """Execute one training stage and return updated *best_val_loss* and *best_metric_score*."""
        # Allow per-stage overrides of batch size / gradient accumulation
        batch_size = stage_batch_size if stage_batch_size is not None else cfg.batch_size
        grad_accum = stage_grad_accum if stage_grad_accum is not None else cfg.gradient_accumulation_steps

        X, Y = get_batch()
        t0 = time.time()
        iter_num = start_iter
        local_iter_num = 0
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num, cfg) * lr_scale
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % cfg.eval_interval == 0:
                losses = estimate_loss(
                    model,
                    ctx,
                    loaders=[("train", train_dataloader), ("val", val_dataloader)],
                    eval_iters=eval_iters,
                )
                if ddp:
                    for key in ["loss/train", "loss/val"]:
                        output = [None] * ddp_world_size
                        th.distributed.all_gather_object(output, losses[key])
                        losses[key] = sum(output) / ddp_world_size
                if master_process:
                    prefix = f"[{stage_label}] " if stage_label else ""
                    logger.info(
                        "{}step {}: train loss {:.4f}, val loss {:.4f}".format(
                            prefix,
                            iter_num,
                            losses["loss/train"],
                            losses["loss/val"],
                        )
                    )
                    if iter_num > 0:
                        checkpoint = {
                            "iter_num": iter_num,
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_val_loss": losses["loss/val"],
                            "best_metric_score": best_metric_score,
                            "model_config": raw_model.config,
                            "vocab": vocab.stoi,
                            "model_type": str(model_type),
                            "wandb_path": wandb_run.path if wandb_run is not None else None,
                            "training_stage": training_stage,
                            "original_num_experts": original_num_experts,
                        }
                        th.save(checkpoint, out_dir / "recent_model.pt")
                        logger.info("Saved the most recent model.")
                        if losses["loss/val"] < best_val_loss:
                            th.save(checkpoint, out_dir / "best_model.pt")
                            logger.info(
                                f"Saved the best model: {best_val_loss} => {losses['loss/val']}"
                            )
                            best_val_loss = losses["loss/val"]

                        if online_logger is not None:
                            stage_tokens_per_iter = (
                                grad_accum * batch_size * cfg.n_positions
                            )
                            epochs = iter_num * stage_tokens_per_iter / len(train_dataset)
                            online_logger.log(
                                {
                                    "other/iter": iter_num,
                                    "other/lr": lr,
                                    "other/mfu": running_mfu * 100,
                                    "other/epochs": epochs,
                                    **losses,
                                }
                            )

            # forward backward update, with optional gradient accumulation
            for micro_step in range(grad_accum):
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum - 1
                    )
                with ctx:
                    if isinstance(X, tuple):
                        output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                    else:
                        output = model(input_ids=X, labels=Y)
                    loss = output.loss
                    # Add MoE auxiliary load-balancing loss if applicable
                    if use_moe and hasattr(output, "moe_loss"):
                        loss = loss + cfg.moe_aux_loss_weight * output.moe_loss
                    loss = (
                        loss / grad_accum
                    )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch
                X, Y = get_batch()
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if cfg.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                th.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % cfg.log_interval == 0 and master_process:
                lossf = loss.item() * grad_accum
                if local_iter_num >= 5 and model_type == ModelType.DECODER:
                    mfu = estimate_mfu(
                        raw_model,
                        num_params,
                        batch_size * grad_accum,
                        dt,
                    )
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                prefix = f"{stage_label} " if stage_label else ""
                logger.info(
                    f"[{prefix}{iter_num}]: loss={lossf:.4f}, "
                    f"time={dt * 1000:.0f}ms, mfu={running_mfu:.2%}"
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters_stage:
                break

        return best_val_loss, best_metric_score

    # ---- Stage 1 (or vanilla training) --------------------------------------
    if not resume_into_stage2:
        stage_label = "Stage 1" if two_stage_moe else ""
        best_val_loss, best_metric_score = run_training_stage(
            model=model,
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            use_moe=use_moe,
            num_params=num_params,
            start_iter=iter_num,
            max_iters_stage=cfg.max_iters,
            best_val_loss=best_val_loss,
            best_metric_score=best_metric_score,
            stage_label=stage_label,
            training_stage=1,
        )
    else:
        if master_process:
            logger.info("Skipping Stage 1 (resuming directly into Stage 2).")

    # ---- Stage 2: MoE fine-tuning (only when two_stage_moe) -----------------
    if two_stage_moe:
        # --- Free Stage 1 model memory before building the larger MoE model ---
        if not resume_into_stage2:
            if master_process:
                logger.info("Stage 1 complete. Saving backbone and beginning Stage 2 (MoE).")
                th.save(
                    {
                        "model": raw_model.state_dict(),
                        "model_config": raw_model.config,
                        "vocab": vocab.stoi,
                    },
                    out_dir / "stage1_backbone.pt",
                )

            # Capture backbone weights before they are overwritten
            stage1_state = {k: v.cpu() for k, v in raw_model.state_dict().items()}

            # Free stage 1 model, optimizer, and compiled wrapper from GPU
            del model, raw_model, optimizer, scaler
            th.cuda.empty_cache() if "cuda" in device else None

        # Restore the original number of experts in cfg
        cfg.num_experts_total = original_num_experts

        # --- Auto-scale batch size / gradient accumulation to avoid OOM ---
        moe_batch_size, moe_grad_accum = _compute_moe_batch_params(
            cfg, original_num_experts, device, master_process
        )

        # Rebuild data loaders with the new (smaller) batch size
        moe_train_dataloader, moe_val_dataloader = (
            DataLoader(
                dataset,
                batch_size=moe_batch_size,
                shuffle=not ddp,
                sampler=DistributedSampler(dataset) if ddp else None,
                num_workers=8,
                pin_memory=_pin_memory,
                persistent_workers=True,
            )
            for dataset in [train_dataset, val_dataset]
        )
        # Replace the infinite loader and val loader used by the closure
        train_dataloader = make_infinite_loader(moe_train_dataloader)
        val_dataloader = moe_val_dataloader
        # Update eval_iters for the new batch size
        eval_iters = len(val_dataset) // (moe_batch_size * cfg.n_positions) + 1

        if resume_into_stage2:
            # Resuming: model already loaded from checkpoint
            moe_raw_model = raw_model
            moe_optimizer_state = optimizer_state
        else:
            # Fresh Stage 2: build MoE model and seed with backbone weights
            moe_raw_model = _build_moe_model(
                cfg, original_num_experts, vocab_size, model_type, train_dataset
            )

            # Load backbone weights — FFN weights are copied into every expert
            moe_state = _convert_to_moe_state_dict(stage1_state, original_num_experts)
            moe_raw_model.load_state_dict(moe_state, strict=False)
            del stage1_state, moe_state

            moe_optimizer_state = None

        # Freeze all non-MoE parameters (attention, embeddings, layer norms, …)
        for name, param in moe_raw_model.named_parameters():
            param.requires_grad = ".moe." in name

        moe_raw_model.to(device)

        moe_use_moe = getattr(moe_raw_model, "use_moe", False)
        # For encoder-decoder, the decoder inside has num_active_parameters;
        # for the wrapper we fall back to num_parameters.
        if hasattr(moe_raw_model, "num_active_parameters"):
            moe_num_params = moe_raw_model.num_active_parameters()
        else:
            moe_num_params = moe_raw_model.num_parameters()
        if master_process:
            n_trainable = sum(p.numel() for p in moe_raw_model.parameters() if p.requires_grad)
            n_frozen = sum(p.numel() for p in moe_raw_model.parameters() if not p.requires_grad)
            logger.info(
                f"Stage 2 — MoE model: active params {moe_num_params / 1e6:.2f}M, "
                f"trainable {n_trainable / 1e6:.2f}M, frozen {n_frozen / 1e6:.2f}M"
            )

        moe_scaler = th.amp.GradScaler(enabled=(cfg.dtype == "float16"))
        moe_optimizer = configure_optimizers(
            moe_raw_model, cfg.weight_decay, cfg.lr * 0.1, (cfg.beta1, cfg.beta2), device
        )
        if moe_optimizer_state is not None:
            moe_optimizer.load_state_dict(moe_optimizer_state)

        if master_process:
            logger.info(("Not c" if cfg.no_compile else "C") + "ompiling the MoE model...")
        moe_model = th.compile(moe_raw_model, disable=cfg.no_compile)
        if ddp:
            moe_model = DDP(moe_model, device_ids=[ddp_local_rank])

        stage2_start_iter = iter_num if resume_into_stage2 else 0
        best_val_loss, best_metric_score = run_training_stage(
            model=moe_model,
            raw_model=moe_raw_model,
            optimizer=moe_optimizer,
            scaler=moe_scaler,
            use_moe=moe_use_moe or (original_num_experts > 1),
            num_params=moe_num_params,
            start_iter=stage2_start_iter,
            max_iters_stage=cfg.max_iters,
            best_val_loss=best_val_loss if resume_into_stage2 else 1e9,
            best_metric_score=best_metric_score if resume_into_stage2 else 0,
            lr_scale=0.1,
            stage_label="Stage 2",
            training_stage=2,
            stage_batch_size=moe_batch_size,
            stage_grad_accum=moe_grad_accum,
        )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
