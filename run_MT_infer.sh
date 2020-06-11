fairseq-generate data-bin/wmt14.en-fr.newstest2014  \
    --path data-bin/wmt14.en-fr.fconv-py/model.pt \
    --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
