from argsparser import ArgsParser
from model import ParakeetModel
from tokenizer import Tokenizer
import torch
from pytorch_lightning import Trainer
import datetime
import logging
from omegaconf import OmegaConf



#region logging
# Logging
# -------
# Set logging config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion


class CustomTrainer:
    def __init__(self, trainer_params) -> None:
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.set_params(trainer_params)
        self.config = OmegaConf.load(self.params.config_file_path)
        self.set_device()
        self.load_tokenizer()
        self.load_network()
        self.generate_model_name()
        self.init_trainer()

    def set_params(self, trainer_params):
        logger.info("Setting Parameters...")
        self.params = trainer_params

        # printing the parameters:
        # we transform the argparse.Namespace() into a dictionary
        params_dict = vars(self.params)

        # we print the dictionary in a sorted way:
        for key, value in sorted(params_dict.items()):
            print(f"{key}: {value}")        

    def load_tokenizer(self):
        logger.info("Loading Tokenizer...")
        self.tokenizer = Tokenizer(self.params)

    def load_network(self):
        logger.info("Loading Network...")
        self.model = ParakeetModel(self.params, self.config, self.device)

    def set_device(self):
        logger.info("Setting the devices...")

        # CUDA things.
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Running on {self.device} device")

        if self.device == "cuda":
            self.gpus_count = torch.cuda.device_count()
            logger.info(f"{self.gpus_count} GPUs available.")
            # Batch size should be divisible by number of GPUs
        else:
            self.gpus_count = 0

        logger.info("Device setted.")
    
    def generate_model_name(self):
        name_components = []

        formatted_datetime = self.start_datetime.replace(':', '_').replace(' ', '_').replace('-', '_')
        name_components.append(formatted_datetime)

        # TODO: add the new components of the name. 

        name_components = [str(component) for component in name_components]
        model_name = "_".join(name_components)

        logger.info(f"The name of the model will be {model_name}")

        self.model_name = model_name


    def init_trainer(self):
        logger.info("Setting up the Pytorch Lightning Trainer")

        logger.info(f"The number of devices is {self.params.devices} the number of nodes {self.params.nodes}")
        
        self.trainer = Trainer(devices=self.params.devices,
                               num_nodes=self.params.nodes,
                               accelerator=self.device,
                               strategy="ddp",
                               max_epochs=self.params.max_epochs,
                               enable_checkpointing=self.params.enable_checkpointing,
                               logger=self.params.logger,
                               log_every_n_steps=self.params.log_every_n_steps,
                               check_val_every_n_epoch=self.params.check_val_every_n_epoch,
                               precision="bf16")
        
        
    def fit_model(self):
        self.trainer.fit(self.model.asr_model)
    
    def save_model(self):
        self.model.asr_model.save_to(self.params.save_directory + self.model_name +".nemo")

    def main(self):
        self.fit_model()
        self.save_model()

def main():
    args_parser = ArgsParser()
    args_parser()
    trainer_params = args_parser.arguments

    custom_trainer = CustomTrainer(trainer_params)
    custom_trainer.main()

if __name__=="__main__":
    main()