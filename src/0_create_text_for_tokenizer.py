import pandas as pd 

train_data = pd.read_csv("/gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/SPANISH_DATALOADER/data/train.tsv", delimiter="\t", 
                            names=["audio_id", "normalized_text", "absolute_path"])
dev_data = pd.read_csv("/gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/SPANISH_DATALOADER/data/dev.tsv", delimiter="\t", 
                            names=["audio_id", "normalized_text", "absolute_path"])

for _, row in train_data.iterrows():
    print(row["normalized_text"])
for _, row in dev_data.iterrows():
    print(row["normalized_text"])