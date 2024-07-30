from omegaconf import OmegaConf, open_dict

class Config:

    def __init__(self, config_file_path):
        self.config = OmegaConf.load(config_file_path)
        self.update_tokenizer_info()
        self.data_processing()
        self.set_dataset_config()
        self.set_warmup_steps()

    def update_tokenizer_info(self):
        self.config.model.tokenizer.dir = TOKENIZER
        self.config.model.tokenizer.update_tokenizer = True
        self.config.model.tokenizer.type = TOKENIZER_TYPE_CFG
        print(OmegaConf.to_yaml(self.config.model.tokenizer))

    def data_processing(self):
        self.config.model.train_ds.normalize_text=False
        self.config.model.normalize_text = True # same as the normalize_text in the get_hf_text_data.py
        self.config.model.symbols_to_keep=["."] # same as the symbols_to_keep in the get_hf_text_data.py
        self.config.model.data_path = 'google/fleurs'
        self.config.model.data_name = 'te_in'
        self.config.model.streaming = False
    
    def set_dataset_config(self):
        self.config.model.train_ds.hf_data_cfg=train_split
        self.config.model.train_ds.text_key='transcription'
        self.config.model.train_ds.batch_size=16 # change this based on your GPU memory.
        self.config.model.train_ds.normalize_text=True

        self.config.model.validation_ds.hf_data_cfg=train_split
        self.config.model.validation_ds.hf_data_cfg.split='validation' # updated this based on the split
        self.config.model.validation_ds.text_key='transcription'
        self.config.model.validation_ds.normalize_text=True

        self.config.model.test_ds.hf_data_cfg=train_split
        self.config.model.test_ds.hf_data_cfg.split='test' # updated this based on the split
        self.config.model.test_ds.text_key='transcription'
        self.config.model.test_ds.normalize_text=True

        print(OmegaConf.to_yaml(self.config.model))
    
    def set_warmup_steps(self):
        self.config.model.optim.sched.warmup_steps = 500 # 10% of the total steps
        self.config.model.optim.lr = 3e-4

        del self.config.model.spec_augment #For this example, we are not using SpecAugment