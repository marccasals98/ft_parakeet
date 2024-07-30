
python scripts/process_asr_text_tokenizer.py \
    --data_file='telugu_train_corpus.txt' \
    --data_root="tokenizers" \
    --tokenizer=$TOKENIZER_TYPE \
    --spe_type=$SPE_TYPE \
    --no_lower_case \
    --log \
    --vocab_size=$VOCAB_SIZE