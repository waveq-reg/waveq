fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path wmt16.en-de.joined-dict.transformer/model.pt \
    --batch-size 128 --beam 5 --remove-bpe
