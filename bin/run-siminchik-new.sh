#!/bin/sh
set -xe

export LD_LIBRARY_PATH=/home/ubuntu/mozillaDeepSpeech/:${LD_LIBRARY_PATH}


if [ ! -d "${COMPUTE_DATA_DIR}" ]; then
    COMPUTE_DATA_DIR="/home/ubuntu/datasets"
fi;

# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/quz-train-new.csv" ]; then
    echo "Warning: It looks like you don't have the Siminchik corpus"       \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the LibriSpeech data is located, and that you ran the" \
         "importer script at bin/import_librivox.py before running this script."
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/siminchik"))')
fi

for train_batch_size in 12 16; do
	for dev_batch_size in 12 16; do
		for test_batch_size in 12 16; do
			for learning_rate in 0.0001 0.001; do
				for display_step in 5 15; do
					for validation_step in 5 15; do
						for dropout_rate in 0.35 0.45; do
                                                        rm -Rf "/home/ubuntu/rodolfo/modelo_final/"
                                                        mkdir "/home/ubuntu/rodolfo/modelo_final/"
							python -u grid_search.py \
							  --train_files "$COMPUTE_DATA_DIR/quz-train-new.csv" \
							  --dev_files "$COMPUTE_DATA_DIR/quz-dev-new.csv" \
							  --test_files "$COMPUTE_DATA_DIR/quz-test-new.csv" \
							  --decoder_library_path "/home/ubuntu/mozillaDeepSpeech/libctc_decoder_with_kenlm.so" \
							  --alphabet_config_path "/home/ubuntu/datasets/quz_alphabet.txt" \
							  --lm_binary_path "/home/ubuntu/quz_data/lm/5-gram.binary" \
							  --lm_trie_path "/home/ubuntu/quz_data/lm/quz_trie" \
							  --train_batch_size ${train_batch_size} \
							  --dev_batch_size ${dev_batch_size} \
							  --test_batch_size ${test_batch_size} \
							  --learning_rate ${learning_rate} \
							  --epoch 15 \
							  --display_step ${display_step} \
							  --validation_step ${validation_step} \
							  --dropout_rate ${dropout_rate} \
							  --default_stddev 0.046875 \
							  --summary_secs 30 \
							  --log_level 0 \
							  --checkpoint_dir "/home/ubuntu/rodolfo/model_final/" \
							  --export_dir "/home/ubuntu/rodolfo/models/"
							  "$@"
                                                         
                                                        rm -Rf "/home/ubuntu/rodolfo/modelo_final/"
						done
					done
				done
			done
		done
	done
done
~  
