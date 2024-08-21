#!/bin/bash
#SBATCH -D .
#SBATCH --account=bsc88                 # Our project account id
#SBATCH -q acc_bscls                    # QoS for life sciences in nodes with GPUs (acc_bscls) / (acc_debug) for debug

#SBATCH --time=0-48:00:00               # acc_bscls wallclock 48h / acc_debug wallclock 2h

#SBATCH -c 20                           # cpus-per-task
#SBATCH --job-name=ft_parakeet_ca_es_upto40_24nodes_4gpus_bs8_Abir
#SBATCH --output=/gpfs/projects/bsc88/speech/ASR/outputs/ft_parakeet/finetuning/%x_%j.log

#SBATCH --gres=gpu:4                    #4
#SBATCH --ntasks-per-node=4             # This needs to match Trainer(devices=...); 4
#SBATCH --nodes=24                       # Number of nodes; 4

#SBATCH --verbose


# Variables for distributed run
# export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export COUNT_GPU="[$(echo $SLURM_JOB_GPUS | cut -d ':' -f 2)]"

echo NODES=$COUNT_NODE
echo GPUS=$COUNT_GPU  # if 2 -> returns 0, 1

srun /gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/shs/2_srun_finetuning.sh