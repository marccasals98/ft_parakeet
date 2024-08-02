from transformers import AutoModel, AutoTokenizer
import os
from nemo.collections.asr.models import ASRModel


# Define the model name and the directory to save the model
model_name = "nvidia/stt_ca_conformer_transducer_large"
save_directory = "./nemo_models"

# Create the directory if it does not exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load the model and tokenizer
asr_model = ASRModel.from_pretrained(model_name)


# Save the model and tokenizer to the directory
asr_model.save_to(save_directory+"/stt_ca_conformer_transducer_large")


print(f"Model and tokenizer saved to {save_directory}")

