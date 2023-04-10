import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

at_biwi = False  # Are you running this code from the ETH Computer Vision Lab (Biwi)?

project_root = '/home/saumgupta/UNet-Zoo'
log_root = '/data/saumgupta/unet-zoo'

dummy_data_root = '/Users/marcgantenbein/PycharmProjects/UNet-Zoo/data'

data_root = '/Users/marcgantenbein/scratch/data/data_lidc.pickle'

uzh_root = '/Users/marcgantenbein/scratch/data/prostate_original.mat'

brats_root = '/Users/marcgantenbein/scratch/data/MICCAI_BraTS_2018_Data_Training_preproc/data_3D_size_128_128_128_res_1.0_1.0_1.0.hdf5'

preproc_folder = '/Users/marcgantenbein/scratch/data/preproc'

uzh_preproc_folder = '/Users/marcgantenbein/scratch/data/preproc'

uzh_input_image_folder = '/Users/marcgantenbein/scratch/data/dummy_uzh/images'
uzh_input_mask_folder = '/Users/marcgantenbein/scratch/data/dummy_uzh/UZH_Prostate_annotations_Christian'