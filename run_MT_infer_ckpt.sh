#!/bin/sh

#mv infer_BLEU.txt infer_BLEU_prev.txt
#touch infer_BLEU.txt

START=59
END=75
STEP=1
for i in $(seq $START $STEP $END)
do
  echo "Looping ... i is set to $i" >> infer_BLEU.txt
  fairseq-generate data-bin/iwslt14.tokenized.de-en \
       --path checkpoints/checkpoint$i.pt \
       --batch-size 128 --beam 5 --remove-bpe
done
