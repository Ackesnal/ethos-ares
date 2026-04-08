#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=sxm
#SBATCH --partition=gpu_sxm
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=960g
#SBATCH --job-name=vanilla_ethos
#SBATCH --account=a_eecs_ds
#SBATCH --time=5-00:00:00
#SBATCH --output=moe_ethos_ares.out
#SBATCH --error=moe_ethos_ares.err

module load cuda/12.6.0
module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate ethos
cd /scratch/user/uqxxu16/ethos-ares/


# this script is intended to be run from the project root
export OMP_NUM_THREADS=64

data_path=meds_data
clear
if [[ ! -d $data_path ]]; then
    echo "Dataset directory not found: $data_path"
    exit 1
fi

model_path=outputs
clear
if [[ ! -d $model_path ]]; then
    echo "Dataset directory not found: $model_path"
    exit 1
fi

BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=8
TOTAL_BATCH_SIZE=$((${BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS} * ${SLURM_GPUS_ON_NODE} * ${SLURM_NNODES}))
N_POSITIONS=2048
N_LAYER=6
N_HEAD=12
N_EMBD=768
N_EXPERTS_TOTAL=4
N_EXPERTS_ACTIVE=2
AUX_LOSS_WEIGHT=0.01
DROPOUT=0.3
LR=5e-3
MIN_LR=1e-6
dataset_name="mimic-iv"
model_name="Layer_${N_LAYER}_Dim_${N_EMBD}_Head_${N_HEAD}_Seq_${N_POSITIONS}_Batch_${TOTAL_BATCH_SIZE}_LR_${LR}_MinLR_${MIN_LR}_Dropout_${DROPOUT}_MoE_${N_EXPERTS_TOTAL}x${N_EXPERTS_ACTIVE}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Get NUM_GPUS from SLURM environment variable
export NUM_GPUS=${SLURM_GPUS_ON_NODE}

mkdir -p "${model_path}/${TIMESTAMP}/${model_name}/"
OUTPUT_FILE="${model_path}/${TIMESTAMP}/${model_name}/training_log.out"
ERROR_FILE="${model_path}/${TIMESTAMP}/${model_name}/training_log.err"

torchrun --no_python --standalone --nproc_per_node=${SLURM_GPUS_ON_NODE} ethos_train \
  data_fp=$data_path/train \
  val_size=6 \
  batch_size=$BATCH_SIZE \
  n_positions=$N_POSITIONS \
  n_layer=$N_LAYER \
  n_head=$N_HEAD \
  n_embd=$N_EMBD \
  num_experts_total=$N_EXPERTS_TOTAL \
  num_experts_activated=$N_EXPERTS_ACTIVE \
  moe_aux_loss_weight=$AUX_LOSS_WEIGHT \
  dropout=$DROPOUT \
  lr=$LR \
  min_lr=$MIN_LR \
  log_interval=10 \
  eval_interval=500 \
  gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  warmup_iters=5000 \
  max_iters=200000 \
  lr_decay_iters=150000 \
  wandb_log=true \
  wandb_project="ethos-meds-$dataset_name" \
  wandb_run_name=$model_name \
  out_dir="${model_path}/${TIMESTAMP}/${model_name}/" \
  > "$OUTPUT_FILE" 2> "$ERROR_FILE"