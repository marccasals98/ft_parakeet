import os
import torch
import datasets
from pytorch_lightning import Trainer
from omegaconf import OmegaConf, open_dict
from nemo.utils import logging, model_utils
from nemo.collections.asr.models import ASRModel
import nemo.collections.asr as nemo_asr
from pytorch_lightning.callbacks import ModelCheckpointom pytorch_lightning import Trainer
from omegaconf import OmegaConf, open_dict
from nemo.utils import logging, model_utils
from nemo.collections.asr.models import ASRModel
import nemo.collections.asr as nemo_asr
from pytorch_lightning.callbacks import ModelCheckpoint


torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
  print("[ INFO ] torch.cuda is available!")
  accelerator = 'gpu'
else:
  accelerator = 'cpu'

from numba import cuda

device = cuda.get_current_device()
print("Max Threads per Block:", device.MAX_THREADS_PER_BLOCK)
print("Max Grid Dimensions:", device.MAX_GRID_DIM_X, device.MAX_GRID_DIM_Y, device.MAX_GRID_DIM_Z)
print("Max Block Dimensions:", device.MAX_BLOCK_DIM_X, device.MAX_BLOCK_DIM_Y, device.MAX_BLOCK_DIM_Z)


train_split = {
            'path': '/gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/CAT-SPA_DATALOADER/loading_script.py',
            'split': 'train', 
            'streaming': False}
dev_split = {
            'path': '/gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/CAT-SPA_DATALOADER/loading_script.py',
            'split': 'dev', 
            'streaming': False}
test_split = {
            'path': '/gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/CAT-SPA_DATALOADER/loading_script.py',
            'split': 'test',
            'streaming': False}

print(OmegaConf.to_yaml(train_split))
print(OmegaConf.to_yaml(dev_split))
print(OmegaConf.to_yaml(test_split))

#-------------- Setting tokenizer params:
VOCAB_SIZE = 256  # can be any value above 29
TOKENIZER_TYPE = "spe"  # can be wpe or spe
SPE_TYPE = "bpe"  # can be bpe or unigram

if TOKENIZER_TYPE == 'spe':
  TOKENIZER = os.path.join("/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/tokenizers", f"tokenizer_spe_{SPE_TYPE}_v{VOCAB_SIZE}")
  TOKENIZER_TYPE_CFG = "bpe"
else:
  TOKENIZER = os.path.join("/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/tokenizers", f"tokenizer_wpe_v{VOCAB_SIZE}")
  TOKENIZER_TYPE_CFG = "wpe"

config = OmegaConf.load("/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/config/speech_to_text_hf_finetune.yaml")
config.model.tokenizer.update_tokenizer = True
config.model.tokenizer.dir = TOKENIZER
config.model.tokenizer.type = TOKENIZER_TYPE_CFG
print("[ INFO ] tokenizer: ", OmegaConf.to_yaml(config.model.tokenizer))

config.model.train_ds.normalize_text=False
config.model.normalize_text = False # same as the normalize_text in the get_hf_text_data.py
#config.model.symbols_to_keep=["."] # same as the symbols_to_keep in the get_hf_text_data.py
config.model.data_path = '/gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/CAT-SPA_DATALOADER/loading_script.py'
config.model.streaming = False

bs = 8
#-------------- Setting training and validation Datasets params:
config.model.train_ds.hf_data_cfg=train_split
config.model.train_ds.text_key='normalized_text'
config.model.train_ds.batch_size=bs
config.model.train_ds.normalize_text=False

config.model.validation_ds.hf_data_cfg=dev_split
config.model.validation_ds.hf_data_cfg.split='validation' 
config.model.validation_ds.text_key='normalized_text'
config.model.validation_ds.batch_size=bs
config.model.validation_ds.normalize_text=False

config.model.test_ds.hf_data_cfg=test_split
config.model.test_ds.hf_data_cfg.split='test'
config.model.test_ds.text_key='normalized_text'
config.model.test_ds.batch_size=bs
config.model.test_ds.normalize_text=False

print(OmegaConf.to_yaml(config.model))

#-------------- Setting Optimization params:
# Current optim looks fine except for the warmup_steps
config.model.optim.sched.warmup_steps = 500 # 10% of the total steps
#config.model.optim.lr = 1e-4
#config.model.optim.lr = 2e-4 # for 16 nodes 
config.model.optim.lr = 2e-4 # for 24 nodes 
print(OmegaConf.to_yaml(config.model.optim))

del config.model.spec_augment #For this example, we are not using SpecAugment

#FROM THE ES PRETRAINED MODEL:
#asr_model=ASRModel.restore_from("/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/models/stt_es_conformer_transducer_large.nemo")

#FROM THE CA PRETRAINED MODEL:
#asr_model = ASRModel.restore_from("/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/models/stt_ca_conformer_transducer_large.nemo")

#FROM PARAKEET:
asr_model = ASRModel.restore_from("/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/nemo_models/parakeet-tdt-1.1b.nemo")

#FOR RESUMING:
#asr_model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint('/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/checkpoints/ca-es/epoch=0-v1.ckpt')

decoder=asr_model.decoder.state_dict()
joint_state = asr_model.joint.state_dict()
prev_vocab_size = asr_model.tokenizer.vocab_size

asr_model.change_vocabulary(new_tokenizer_dir=TOKENIZER, new_tokenizer_type=TOKENIZER_TYPE_CFG)
if asr_model.tokenizer.vocab_size == prev_vocab_size: # checking new tokenizer vocab size
    asr_model.decoder.load_state_dict(decoder)
    asr_model.joint.load_state_dict(joint_state)

#-------------- Loading data loaders:
cfg = model_utils.convert_model_config_to_dict_config(config)
asr_model.setup_training_data(cfg.model.train_ds)
asr_model.setup_validation_data(cfg.model.validation_ds)

if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
    asr_model.setup_test_data(cfg.model.test_ds)

# We will also reduce the hidden dimension of the joint and the prediction networks to preserve some memory
asr_model.setup_optimization(cfg.model.optim)

#-------------- Checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath="/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/checkpoints/ca-es/parakeet/",
    filename='{epoch}',
    every_n_epochs=1,
    save_top_k=-1
)
#-------------- Initialize a Trainer for the Transducer model
trainer = Trainer(
                devices=4, #4
                num_nodes=24, #4
                accelerator=accelerator, 
                strategy="ddp",
                max_epochs=10,
                enable_checkpointing=True, #set this to True to have the checkpoints
                logger=True, #set this to True to have lightning_logs
                log_every_n_steps=100, 
                check_val_every_n_epoch=1, 
                precision='bf16',
                callbacks=[checkpoint_callback]
                )

#-------------- Build the model
trainer.fit(asr_model)
asr_model.save_to("/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_conformer_transducer/checkpoints/ca-es/parakeet/ca_es_model_parakeet.nemo")