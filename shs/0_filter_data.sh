export HYDRA_FULL_ERROR=1;python src/get_hf_text_data.py \
    --config-path="../configs" \
    --config-name="huggingface_data_tokenizer" \
    normalize_text=True \
    symbols_to_keep=["."] \
    text_key="transcription" \
    output_file='telugu_train_corpus.txt' \
    +hf_data_cfg='[{train_split}]'