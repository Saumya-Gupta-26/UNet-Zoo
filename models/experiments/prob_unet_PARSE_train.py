import torch
import torch.nn as nn
from models.probabilistic_unet3D import ProbabilisticUnet3D
from utils import normalise_image

dataname = 'parse'
experiment_name = 'ProbabilisticUnet_PARSE'
log_dir_name = 'PARSE'

folders = ["/data/saumgupta/parse2022/nnUNet_raw_data/Task101_VESSEL/imagesTr", "/data/saumgupta/parse2022/nnUNet_raw_data/Task101_VESSEL/labelsTr"]
train_datalist = "/home/saumgupta/dmt-crf-gnn-mlp/3D/datalists/sbu/train-list.csv"
validation_datalist = "/home/saumgupta/dmt-crf-gnn-mlp/3D/datalists/sbu/val-list.csv"

# number of filter for the latent levels, they will be applied in the order as loaded into the list
filter_channels = [32, 64, 128]
latent_levels = 1
latent_dim = 6

iterations = 5000000

n_classes = 2
num_labels_per_subject = 1 # saum: i think this is how many annotations per sample (multi-annotators case)

no_convs_fcomb = 3 # not used
beta = 1.0 # not used

use_reversible = False
exponential_weighting = True

# use 1 for grayscale (eg ROSE, parse), 3 for RGB images (eg DRIVE)
input_channels = 1
epochs_to_train = 20
train_batch_size = 1
val_batch_size = 1
image_size = (1, 128, 128, 128)

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': n_classes}

input_normalisation = normalise_image

validation_samples = 4
num_validation_images = 4

logging_frequency = 1000
validation_frequency = 1000

weight_decay = 10e-5
learning_rate = 1e-4

pretrained_model = None
# model
model = ProbabilisticUnet3D


# # number of filter for the latent levels, they will be applied in the order as loaded into the list
# filter_channels = [32, 64, 128, 192]
# latent_levels = 1 # TODO: this is passed to latent dim and latent levels, should not be like that
# resolution_level = 7
#
# n_classes = 2
# no_convs_fcomb = 4
# beta = 10.0 # for loss function
# #
# use_reversible = False
#
# # use 1 for grayscale, 3 for RGB images
# input_channels = 1
#
# epochs_to_train = 20
# batch_size = [12, 1, 1]
#
# validation_samples = 16
#
# logging_frequency = 10
# validation_frequency = 100
#
# input_normalisation = normalise_image
#
# # model
# model = ProbabilisticUnet
