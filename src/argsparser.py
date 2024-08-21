import argparse
from omegaconf import OmegaConf

class ArgsParser:

    def __init__(self) -> None:
        self.initialize_parser()

    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Train Parakeet ASR model',
            )
    
    def add_parser_args(self,cfg):
        self.parser.add_argument(
            '--parakeet_flavour',
            type=str,
            default=cfg.train_settings.parakeet_flavour,
            choices=["parakeet-rnnt-0.6b.nemo", "parakeet-0.6b", "parakeet-tdt-1.1b.nemo"]
        )

        self.parser.add_argument(
            '--config_file_path',
            type=str,
            default=cfg.train_settings.config_file_path 
        )
        self.parser.add_argument(
            '--vocab_size',
            type=str,
            default=cfg.train_settings.vocab_size 
        )
        self.parser.add_argument(
            '--tokenizer_dir',
            type=str,
            default=cfg.train_settings.tokenizer_dir
        )

        self.parser.add_argument(
            '--tokenizer_type',
            type=str,
            default=cfg.train_settings.tokenizer_type 
        )
        self.parser.add_argument(
            '--spe_type',
            type=str,
            default=cfg.train_settings.spe_type
        )             
        self.parser.add_argument(
            '--max_steps',
            type=str,
            default=cfg.train_settings.max_steps
        )
        self.parser.add_argument(
            '--max_epochs',
            type=str,
            default=cfg.train_settings.max_epochs
        )                
        self.parser.add_argument(
            '--devices',
            type=int,
            default=cfg.train_settings.devices
        )
        self.parser.add_argument(
            '--nodes',
            type=int,
            default=cfg.train_settings.nodes
        )        
        self.parser.add_argument(
            '--enable_checkpointing',
            type=str,
            default=cfg.train_settings.enable_checkpointing
        )                
        self.parser.add_argument(
            '--logger',
            type=str,
            default=cfg.train_settings.logger
        )                
        self.parser.add_argument(
            '--log_every_n_steps',
            type=str,
            default=cfg.train_settings.log_every_n_steps
        )                
        self.parser.add_argument(
            '--check_val_every_n_epoch',
            type=str,
            default=cfg.train_settings.check_val_every_n_epoch
        )

        self.parser.add_argument(
            '--save_directory',
            type=str,
            default=cfg.train_settings.save_directory
        )
        self.parser.add_argument(
            '--pretrained_dir',
            type=str,
            default=cfg.train_settings.pretrained_dir
        )                

    def main(self):
        cfg = OmegaConf.load("/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_parakeet/configs/basic_training.yaml")
        self.add_parser_args(cfg)
        self.arguments = self.parser.parse_args()

    def __call__(self):
        self.main()     


