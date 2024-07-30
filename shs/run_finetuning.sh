#!/bin/bash
#SBATCH -D .
#SBATCH --account=bsc88                 # Our project account id
#SBATCH -q acc_debug                   # QoS for life sciences in nodes with GPUs (acc_bscls) / (acc_debug) for debug

#SBATCH --time=0-02:00:00               # acc_bscls wallclock 48h / acc_debug wallclock 2h
#SBATCH -c 20                           # cpus-per-task
#SBATCH --job-name=ft_parakeet_1node_1Gpu
#SBATCH --output=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/logs/%x_job-%j.log

#SBATCH --gres=gpu:1                    #4
#SBATCH --ntasks-per-node=1            # This needs to match Trainer(devices=...); 4
#SBATCH --nodes=1                      # Number of nodes; 4

#SBATCH --verbose

echo "Task per node $SLURM_NTASKS_PER_NODE nodes $SLURM_NNODES "

source /gpfs/projects/bsc88/speech/ASR/environments/nemo_050224/bin/activate

# HACK: I will hardcode some varaibles that will be used in the pipepline: -----------

#TOKENIZER VARIABLES
VOCAB_SIZE=256  # can be any value above 29
TOKENIZER_TYPE="spe"  # can be wpe or spe
SPE_TYPE="bpe"  # can be bpe or unigram
DATA_FILE="/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/data/text_for_tokenizer.txt"

#FINE-TUNING VARIABLES
HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES="0,1,2,3"
INIT_MODEL=parakeet-rnnt-0.6b.nemo
RESULTS="/gpfs/projects/bsc88/speech/ASR/outputs/ft_parakeet/"

# ---------------------------------------------------------------------------------------

# 0 select data:
# select the datasets in the list of this folder /gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/CAT-SPA_DATALOADER/0_Select_Datasets

python /gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/CAT-SPA_DATALOADER/loading_script.py

# 1 Create tokenizer:
echo "Creating the tokenizer..."
python /gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/src/1_process_asr_text_tokenizer.py \
    --data_file=$DATA_FILE \
    --data_root="tokenizers" \
    --tokenizer=$TOKENIZER_TYPE \
    --spe_type=$SPE_TYPE \
    --no_lower_case \
    --log \
    --vocab_size=$VOCAB_SIZE

echo "Tokenizer Created!"

# 2 Fine-Tuning:
echo "Starting the fine-tuning"
export TORCH_HOME=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/cache/ca-es
export HF_MODULES_CACHE=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/cache/ca-es
export HF_DATASETS_CACHE=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/cache/ca-es
export HF_HOME=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/cache/ca-es
export NEMO_CACHE_DIR=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/cache/ca-es

export PYTHONFAULTHANDLER=1


HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES="0,1,2,3"

export NCCL_DEBUG=info
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

#Additional flags
# ENV VARIABLES
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SLURM_NTASKS_PER_NODE=4

### DVD BSC SUPPORT VARIABLES
export NCCL_IB_TIMEOUT=70
export NCCL_IB_HCA=mlx5
export NCCL_IB_RETRY_CNT=40
export WANDB_HTTP_TIMEOUT=8000
export WANDB_INIT_TIMEOUT=8000


python /gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/src/train.py \
    --devices=$SLURM_NTASKS_PER_NODE \
    --nodes=$SLURM_NNODES \
    --parakeet_flavour="parakeet-rnnt-0.6b.nemo"
