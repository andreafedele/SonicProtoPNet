import os
import time
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir, get_std_dev
import model
import push
# import prune
import train_and_test as tnt
import save
from log import create_logger
# from preprocess import mean, std, preprocess_input_function

import torchaudio
from audio_dataset import AudioDataset

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
# print(os.environ['CUDA_VISIBLE_DEVICES'])

torch.cuda.set_per_process_memory_fraction(0.5, torch.device('cuda:0'))

# book keeping namings and code
from settings import base_architecture, img_channels, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, train_annotation_dir, test_annotation_dir, train_push_annotation_dir, train_batch_size, test_batch_size, train_push_batch_size, \
    sample_rate, num_samples, n_fft, hop_length, n_mels, power_or_db

mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)

cut_dimensions = (n_mels, n_mels)
img_size = n_mels # img_size to initiate ppnet depends on n_mels dimension (must be a square)

# train dataset
train_dataset = AudioDataset(train_annotation_dir, train_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False
)
print(f"There are {len(train_dataset)} samples in the train dataset.")

# get dimensions from one training sample, so to cut augumented signals if necessary
# ss, _ = train_dataset[0]


# train push dataset (train augmented)
train_push_dataset = AudioDataset(train_push_annotation_dir, train_push_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions) 
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False
)
print(f"There are {len(train_push_dataset)} samples in the train push (augmented) dataset.")

# test dataset
test_dataset = AudioDataset(test_annotation_dir, test_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions)
print(f"There are {len(test_dataset)} samples in the test dataset.")
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False
)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# get the start time
st = time.time()

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type,
                              img_channels=img_channels)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr, last_layer_convex_optimizations
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# early stopping on training set parameters
from settings import es_last_n_epochs, es_conv_threshold, target_accu
training_acc_converged = False
clust_cost_smaller_than_sep_cost = False

# train the model
log('start training')
train_accs = []
# import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, _obj = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer, class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        acc, _obj = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer, class_specific=class_specific, coefs=coefs, log=log)
        train_accs.append(acc) # appending training accuracy 
        clust_cost_smaller_than_sep_cost = True if _obj['cluster'] < _obj['separation'] else False
        joint_lr_scheduler.step()

    accu, _obj = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu, target_accu=target_accu, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=None, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        
        accu, _obj = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu, target_accu=target_accu, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(last_layer_convex_optimizations):
                log('iteration: \t{0}'.format(i))
                _, _obj = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, class_specific=class_specific, coefs=coefs, log=log)
                accu, _obj = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu, target_accu=target_accu, log=log)
    
    if epoch >= es_last_n_epochs + num_warm_epochs:
        print(train_accs)
        print(train_accs[len(train_accs)-es_last_n_epochs:len(train_accs)])
        std_dev = get_std_dev(train_accs[len(train_accs)-es_last_n_epochs:len(train_accs)])
        training_acc_converged = True if std_dev < es_conv_threshold else False

    if training_acc_converged and clust_cost_smaller_than_sep_cost:
        log('Early stop training because: (1) training acc convergend && (2) cluster cost is smaller then separation cost on training set')
        accu, _obj = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu, target_accu=target_accu, log=log)

        break


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
log('Execution time (minutes): ' + str(elapsed_time / 60))

logclose()

