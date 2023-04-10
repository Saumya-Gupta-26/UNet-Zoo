'''
Saumya: Making modifications to use custom datasets

run command:

CUDA_VISIBLE_DEVICES=6 python test_model_saum.py /home/saumgupta/UNet-Zoo/models/experiments/prob_unet_DRIVE_test.py local
CUDA_VISIBLE_DEVICES=7 python test_model_saum.py /home/saumgupta/UNet-Zoo/models/experiments/phiseg_7_5_6_DRIVE_test.py local
'''
import os
from importlib.machinery import SourceFileLoader
import argparse

import utils

# if not sys_config.running_on_gpu_host:
#     import matplotlib.pyplot as plt
import torch
from train_model import UNetModel
from dataloader import DRIVE_folder
import shutil
from data.lidc_data import lidc_data
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

#structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    parser.add_argument("LOCAL", type=str, help="Is this script run on the local machine or the BIWI cluster?")
    args = parser.parse_args()

    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')

    if args.LOCAL == 'local':
        print('Running with local configuration')
        import config.local_config as sys_config
        import matplotlib.pyplot as plt
    else:
        import config.system as sys_config

    exp_config = SourceFileLoader(config_module, config_file).load_module()

    log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)

    utils.makefolder(log_dir)

    shutil.copy(exp_config.__file__, log_dir)

    basic_logger = utils.setup_logger('basic_logger', log_dir + '/test_log_lowest_ged.log')

    basic_logger.info('Running experiment with script: {}'.format(config_file))

    basic_logger.info('!!!! Copied exp_config file to experiment folder !!!!')

    basic_logger.info('**************************************************************')
    basic_logger.info(' *** Running Experiment: %s', exp_config.experiment_name)
    basic_logger.info('**************************************************************')

    model = UNetModel(exp_config, logger=basic_logger, tensorboard=False)
    transform = None

    #data = exp_config.data_loader(sys_config=sys_config, exp_config=exp_config)
    test_set = DRIVE_folder(exp_config.train_datalist, exp_config.folders)
    test_generator = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False,num_workers=1, drop_last=False)

    model.test(test_generator, sys_config=sys_config)
    model.generate_images(test_generator, sys_config=sys_config)


