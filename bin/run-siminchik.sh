#!/bin/sh
set -xe

export LD_LIBRARY_PATH=/home/ubuntu/mozillaDeepSpeech/:${LD_LIBRARY_PATH}

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -d "${COMPUTE_DATA_DIR}" ]; then
    COMPUTE_DATA_DIR="/home/ubuntu/datasets"
fi;

# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/quz-train.csv" ]; then
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

python -u DeepSpeech.py \
  --train_files "$COMPUTE_DATA_DIR/quz-train.csv" \
  --dev_files "$COMPUTE_DATA_DIR/quz-dev.csv" \
  --test_files "$COMPUTE_DATA_DIR/quz-test.csv" \
  --decoder_library_path "/home/ubuntu/mozillaDeepSpeech/libctc_decoder_with_kenlm.so" \
  --alphabet_config_path "/home/ubuntu/datasets/quz_alphabet.txt" \
  --lm_binary_path "/home/ubuntu/quz_data/lm/5-gram.binary" \
  --lm_trie_path "/home/ubuntu/quz_data/lm/quz_trie" \
  --train_batch_size 12 \
  --dev_batch_size 12 \
  --test_batch_size 12 \
  --learning_rate 0.0001 \
  --epoch 20 \
  --display_step 5 \
  --validation_step 5 \
  --dropout_rate 0.30 \
  --default_stddev 0.046875 \
  --summary_secs 30 \
  --log_level 0 \
  --checkpoint_dir "$checkpoint_dir" \
  --export_dir "/home/ubuntu/quz_data/models/"
  "$@"
~  
