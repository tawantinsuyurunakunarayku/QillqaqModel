from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# Importer
# ========

tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')
tf.app.flags.DEFINE_boolean ('fulltrace',        False,       'if full trace debug info should be generated during training')

# Cluster configuration
# =====================

tf.app.flags.DEFINE_string  ('ps_hosts',         '',          'parameter servers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('worker_hosts',     '',          'workers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('job_name',         'localhost', 'job name - one of localhost (default), worker, ps')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of task within the job - worker with index 0 will be the chief')
tf.app.flags.DEFINE_integer ('replicas',         -1,          'total number of replicas - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('replicas_to_agg',  -1,          'number of replicas to aggregate - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('coord_retries',    100,         'number of tries of workers connecting to training coordinator before failing')
tf.app.flags.DEFINE_string  ('coord_host',       'localhost', 'coordination server host')
tf.app.flags.DEFINE_integer ('coord_port',       2500,        'coordination server port')
tf.app.flags.DEFINE_integer ('iters_per_worker', 1,           'number of train or inference iterations per worker before results are sent back to coordinator')

# Global Constants
# ================

tf.app.flags.DEFINE_boolean ('train',            True,        'whether to train the network')
tf.app.flags.DEFINE_boolean ('test',             True,        'whether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            75,          'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'whether to use GPU bound Warp-CTC')

tf.app.flags.DEFINE_float   ('dropout_rate',     0.05,        'dropout rate for feedforward layers')
tf.app.flags.DEFINE_float   ('dropout_rate2',    -1.0,        'dropout rate for layer 2 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate3',    -1.0,        'dropout rate for layer 3 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate4',    0.0,         'dropout rate for layer 4 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate5',    0.0,         'dropout rate for layer 5 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate6',    -1.0,        'dropout rate for layer 6 - defaults to dropout_rate')

tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters

tf.app.flags.DEFINE_float   ('beta1',            0.9,         'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('beta2',            0.999,       'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.001,       'learning rate of Adam optimizer')

# Batch sizes

tf.app.flags.DEFINE_integer ('train_batch_size', 1,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,           'number of elements in a test batch')

# Sample limits

tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set - 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set- 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set- 0 means no limit')

# Step widths

tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model - a detailed progress report is dependent on "--display_step" - 0 means no validation steps')

# Checkpointing

tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  600,         'checkpoint saving interval in seconds')
tf.app.flags.DEFINE_integer ('max_to_keep',      5,           'number of checkpoint files to keep - default value is 5')

# Exporting

tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'whether to remove old exported models')
tf.app.flags.DEFINE_boolean ('use_seq_length',   True,        'have sequence_length in the exported graph (will make tfcompile unhappy)')

# Reporting

tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')

tf.app.flags.DEFINE_string  ('wer_log_pattern',  '',          'pattern for machine readable global logging of WER progress; has to contain %%s, %%s and %%f for the set name, the date and the float respectively; example: "GLOBAL LOG: logwer(\'12ade231\', %%s, %%s, %%f)" would result in some entry like "GLOBAL LOG: logwer(\'12ade231\', \'train\', \'2017-05-18T03:09:48-0700\', 0.05)"; if omitted (default), there will be no logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'whether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     0,           'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')

# Geometry

tf.app.flags.DEFINE_integer ('n_hidden',         2048,        'layer width to use when initialising layers')

# Initialization

tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.046875,    'default standard deviation to use when initialising weights and biases')

# Early Stopping

tf.app.flags.DEFINE_boolean ('early_stop',       True,        'enable early stopping mechanism over validation dataset. Make sure that dev FLAG is enabled for this to work')

# This parameter is irrespective of the time taken by single epoch to complete and checkpoint saving intervals.
# It is possible that early stopping is triggered far after the best checkpoint is already replaced by checkpoint saving interval mechanism.
# One has to align the parameters (earlystop_nsteps, checkpoint_secs) accordingly as per the time taken by an epoch on different datasets.

tf.app.flags.DEFINE_integer ('earlystop_nsteps',  4,          'number of steps to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
tf.app.flags.DEFINE_float   ('estop_mean_thresh', 0.5,        'mean threshold for loss to determine the condition if early stopping is required')
tf.app.flags.DEFINE_float   ('estop_std_thresh',  0.5,        'standard deviation threshold for loss to determine the condition if early stopping is required')

# Decoder

tf.app.flags.DEFINE_string  ('decoder_library_path', 'native_client/libctc_decoder_with_kenlm.so', 'path to the libctc_decoder_with_kenlm.so library containing the decoder implementation.')
tf.app.flags.DEFINE_string  ('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
tf.app.flags.DEFINE_string  ('lm_binary_path',       'data/lm/lm.binary', 'path to the language model binary file created with KenLM')
tf.app.flags.DEFINE_string  ('lm_trie_path',         'data/lm/trie', 'path to the language model trie file created with native_client/generate_trie')
tf.app.flags.DEFINE_integer ('beam_width',        1024,       'beam width used in the CTC decoder when building candidate transcriptions')
tf.app.flags.DEFINE_float   ('lm_weight',         1.75,       'the alpha hyperparameter of the CTC decoder. Language Model weight.')
tf.app.flags.DEFINE_float   ('word_count_weight', 1.00,      'the beta hyperparameter of the CTC decoder. Word insertion weight (penalty).')
tf.app.flags.DEFINE_float   ('valid_word_count_weight', 1.00, 'valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary.')

# Inference mode

tf.app.flags.DEFINE_string  ('one_shot_infer',       '',       'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it. Disables training, testing and exporting.')

# Initialize from frozen model

tf.app.flags.DEFINE_string  ('initialize_from_frozen_model', '', 'path to frozen model to initialize from. This behaves like a checkpoint, loading the weights from the frozen model and starting training with those weights. The optimizer parameters aren\'t restored, so remember to adjust the learning rate.')

for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)

FLAGS = tf.app.flags.FLAGS
