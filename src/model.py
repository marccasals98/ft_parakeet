from nemo.collections.asr.models import ASRModel
import nemo.collections.asr as nemo_asr
from nemo.utils import logging, model_utils
import os


class ParakeetModel:

    def __init__(self, params, config, device)->None:

        self.init_model(params)
        self.update_decoder(params, config)
        self.load_dataloader(params, config)
    
    def init_model(self, params):
        """
        This function imports the model. If there is no Internet some modifications will need to made. 
        """
        model_path = os.path.join(params.pretrained_dir,params.parakeet_flavour)
        print(f"The path of the model is {model_path}")
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)

        
    def update_decoder(self, params, config):
        """
        After loading the pretrained model, decoder of the model needs to updated to support the new language.
        """
        decoder = self.asr_model.decoder.state_dict()
        joint_state = self.asr_model.joint.state_dict()
        prev_vocab_size = self.asr_model.tokenizer.vocab_size

        self.asr_model.change_vocabulary(new_tokenizer_dir=params.tokenizer_dir, new_tokenizer_type=params.spe_type)
        if self.asr_model.tokenizer.vocab_size == prev_vocab_size: # checking new tokenizer vocab size
            self.asr_model.decoder.load_state_dict(decoder)
            self.asr_model.joint.load_state_dict(joint_state)
    
    def load_dataloader(self, params, config):
        cfg = model_utils.convert_model_config_to_dict_config(config)
        self.asr_model.setup_training_data(cfg.model.train_ds)
        self.asr_model.setup_validation_data(cfg.model.validation_ds)
        if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
            self.asr_model.setup_test_data(cfg.model.test_ds)