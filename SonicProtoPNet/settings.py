base_architecture = 'vgg19'
img_channels = 1
prototype_shape = (500, 128, 1, 1)
num_classes = 50
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001_esc50'

data_path = '../datasets/esc50_split/'
train_dir = data_path + 'train_augmented/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'
train_annotation_dir = data_path + 'annotations_train_augmented.csv'
test_annotation_dir = data_path + 'annotations_test.csv'
train_push_annotation_dir = data_path + 'annotations_train.csv'
train_batch_size = 27
test_batch_size = 30
train_push_batch_size = 25

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 5
push_epochs = [i for i in range(num_train_epochs) if i % 5 == 0]
last_layer_convex_optimizations = 5 # 20 si hard-coded by default in ProtoPNet

target_accu = 0.50

# --- train early stopping ---
es_last_n_epochs = 5 # last n epochs to watch in order to verify if convergence was reached on train accuracy
es_conv_threshold = 0.01 # convergency threshold, if std(acc(last_n_epochs)) < thr then convergence has been reached

# --- audio input data-type integration ---
# audio sample rate
sample_rate = 41000
num_samples = 41000

# spectrogram conversion
n_fft = 4096 * 3
hop_length = 600
n_mels = 300

# power spectrogram or dB units spect
power_or_db = 'd' # power spectrogram 'p', decibel dB units 'd'
