#!/bin/bash 

############### LOGGING ###############

############### Load modules ###############
module load cuda/12.1
############### Activate venv ###############
source /gpfs/projects/bsc88/speech/ASR/environments/nemo_050224/bin/activate

export TORCH_HOME=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/cache/ca-es
export HF_MODULES_CACHE=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/cache/ca-es
export HF_DATASETS_CACHE=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/cache/ca-es
export HF_HOME=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/cache/ca-es
export NEMO_CACHE_DIR=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/cache/ca-es

HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES="0,1,2,3"

#INIT_MODEL=stt_es_conformer_transducer_large
RESULTS="/gpfs/projects/bsc88/speech/ASR/outputs/ft_conformer_transducer/"

python /gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/src/2_finetuning_Abir.py \
    --config-path=/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/config/ \
    --config-name=speech_to_text_hf_finetune \
    #+init_from_pretrained_model=${INIT_MODEL} \
    #exp_manager.name="finetuning_catalan_spanish_model" \
    exp_manager.exp_dir=$RESULTS/ \
    exp_manager.create_tensorboard_logger=true \
    exp_manager.create_checkpoint_callback=true \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    #exp_manager.version="version_4432818"
    