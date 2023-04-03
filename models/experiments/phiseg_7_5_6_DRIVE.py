import torch
import torch.nn as nn
from models.phiseg import PHISeg
from utils import normalise_image

experiment_name = 'PHISeg_7_5_6_DRIVE'
log_dir_name = 'DRIVE'

folders = ["/data/saumgupta/DRIVE/training/images", "/data/saumgupta/DRIVE/training/1st_manual"]
train_datalist = "/home/saumgupta/dmt-crf-gnn-mlp/2D/params/train-list.csv"
validation_datalist = "/home/saumgupta/dmt-crf-gnn-mlp/2D/params/val-list.csv"

# number of filter for the latent levels, they will be applied in the order as loaded into the list
filter_channels = [32, 64, 128, 192, 192, 192, 192]
latent_levels = 5

iterations = 5000000

n_classes = 2
num_labels_per_subject = 1 # saum: i think this is how many annotations per sample (multi-annotators case)

no_convs_fcomb = 4 # not used
beta = 10.0 # not used
#
use_reversible = False
exponential_weighting = True

# use 1 for grayscale, 3 for RGB images
input_channels = 3
epochs_to_train = 20
train_batch_size = 6
val_batch_size = 1
image_size = (1, 128, 128)

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': n_classes}

input_normalisation = normalise_image

validation_samples = 2
num_validation_images = 2

logging_frequency = 1000
validation_frequency = 1000

weight_decay = 10e-5
learning_rate = 1e-3

pretrained_model = None #'PHISeg_best_ged.pth'
# model
model = PHISeg
