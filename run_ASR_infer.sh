DIR_FOR_PREPROCESSED_DATA=/backup/speech_data/librispeech_final/
SET=test-clean
RES_DIR=results_dir
MODEL_PATH=model_save

python examples/speech_recognition/infer.py \
	$DIR_FOR_PREPROCESSED_DATA \
	--task speech_recognition \
	--max-tokens 25000 \
	--nbest 1 \
	--path $MODEL_PATH/checkpoint_last.pt \
	--beam 20 \
	--results-path $RES_DIR \
	--batch-size 40 \
	--gen-subset $SET \
	--user-dir examples/speech_recognition/

