# load all into cpu
# do cropping to patch size
# restrict number of slices (see null slices)
# normalize range
# test code by outputting few patches
# training, testing, val
import torch
import os, glob, sys
import numpy as np
from PIL import Image
from os.path import join as pjoin
from torchvision import transforms
from torch.utils import data
from skimage import io

class CREMI(data.Dataset):
    def __init__(self, listpath, filepaths, is_training=False):
    
        self.listpath = listpath
        self.imgfile = filepaths[0]
        self.gtfile = filepaths[1]

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []

        self.indices = []
        self.crop_size = 128
        self.is_training = is_training

        self.loadCPU()

    def loadCPU(self):
        
        img = io.imread(self.imgfile)
        gt = io.imread(self.gtfile) 

        _, h,w = np.shape(img)
        img_tiff = np.zeros((h,w))
        gt_tiff = np.zeros((h,w))
        img_tiff = np.array(img)
        gt_tiff = 1 - np.array(gt)/255

        img = torch.tensor(img_tiff)
        gt = torch.tensor(gt_tiff)   

        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist): 

            meanval = torch.Tensor.float(img[int(entry)]).mean()
            stdval = torch.Tensor.float(img[int(entry)]).std()
            
            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append((img[int(entry)] - meanval) / stdval)
            self.dataCPU['label'].append(gt[int(entry)])


    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.is_training is True:
            # crop to patchsize. compute top-left corner first
            H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)
            torch_img = torch_img[corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
        # else:
        #     torch_img = torch_img[0:1248, 0:1248]
        #     torch_gt = torch_gt[0:1248, 0:1248]

        torch_img = torch.unsqueeze(torch_img,dim=0).repeat(1,1,1)
        torch_gt = torch.unsqueeze(torch_gt,dim=0)
        return torch_img, torch_gt 

class ISBI2013(data.Dataset):
    def __init__(self, listpath, filepaths, is_training=False):
    
        self.listpath = listpath
        self.imgfile = filepaths[0]
        self.gtfile = filepaths[1]

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []

        self.indices = []
        self.crop_size = 128
        self.is_training = is_training

        self.loadCPU()

    def loadCPU(self):
        
        img = io.imread(self.imgfile)
        gt = io.imread(self.gtfile) 

        _, h,w = np.shape(img)
        img_tiff = np.zeros((h,w))
        gt_tiff = np.zeros((h,w))
        img_tiff = np.array(img)
        gt_tiff = 1 - np.array(gt)/255

        img = torch.tensor(img_tiff)
        gt = torch.tensor(gt_tiff)   

        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist): 

            meanval = torch.Tensor.float(img[int(entry)]).mean()
            stdval = torch.Tensor.float(img[int(entry)]).std()

            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append((img[int(entry)] - meanval) / stdval)
            self.dataCPU['label'].append(gt[int(entry)])


    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.is_training is True:
            # crop to patchsize. compute top-left corner first
            H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)
            torch_img = torch_img[corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

        torch_img = torch.unsqueeze(torch_img,dim=0).repeat(1,1,1)
        torch_gt = torch.unsqueeze(torch_gt,dim=0)
        return torch_img, torch_gt 

class DRIVE(data.Dataset):
    def __init__(self, listpath, folderpaths, is_training=False, is_testing=False, constcorner=False):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.indices = [] 
        self.to_tensor = transforms.ToTensor()
        self.crop_size = 128
        self.is_training = is_training
        self.is_testing = is_testing
        self.constcorner = constcorner

        self.loadCPU()

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist):
            components = entry.split('.')
            filename = components[0]

            if self.is_testing:
                im_path = pjoin(self.imgfolder, filename) + '_test.tif'
            else:
                im_path = pjoin(self.imgfolder, filename) + '_training.tif'

            gt_path = pjoin(self.gtfolder, filename) + '_manual1.gif'

            img = Image.open(im_path)
            gt = Image.open(gt_path)

            img = self.to_tensor(img)
            gt = self.to_tensor(gt)

            #normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval

            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['filename'].append(str(filename))

    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.is_training is True:
            # crop to patchsize. compute top-left corner first
            # have a hunch that dmt cannot handle completely black image. so retry if GT has no structure --- this is hoping network outputs something when GT is non-black. 
            # Actually, we cannot have multiple DMT trainings in parallel because they access the same dipha txt files
            C, H, W = torch_img.shape
            #while True:
            if self.constcorner:
                corner_h = 128
                corner_w = 118
            else:
                corner_h = np.random.randint(low=0, high=H-self.crop_size)
                corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

            #print(corner_h, corner_w)

                #if torch.sum(new_torch_gt)  != 0. :
                #    break
                #else:
                #    print("Black GT; Retrying patch generateion...")

        #torch_img = new_torch_img
        #torch_gt = new_torch_gt

        return torch_img, torch_gt, self.dataCPU['filename'][index]


class DRIVE_folder(data.Dataset):
    def __init__(self, listpath, folderpaths):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.suffix = ".tif"

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.indices = [] 
        self.to_tensor = transforms.ToTensor()

        self.loadCPU()

    def loadCPU(self):
        mylist = glob.glob(self.imgfolder + "/*" + self.suffix)
        subdir = False
        if len(mylist) == 0:
            subdir = True
            mylist = glob.glob(self.imgfolder + "/*/*" + self.suffix)

        assert len(mylist) != 0

        mylist.sort()        

        for i, im_path in enumerate(mylist):
            #gt_path = pjoin(self.gtfolder, filename) + '_manual1.gif'
            fname = im_path.replace(self.suffix, ".png").split('/')[-1]
            fname = "gt_" + im_path.replace("_test", "").split('/')[-2] + '/' + fname
            gt_path = glob.glob(self.gtfolder + "/" + fname)

            assert len(gt_path) == 1
            gt_path = gt_path[0]

            img = Image.open(im_path)
            gt = np.array(Image.open(gt_path))[:,:,0]/255.

            img = self.to_tensor(img)
            gt = torch.from_numpy(gt)

            #normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval

            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['filename'].append(im_path.split('/')[-2] + '__' + im_path.split('/')[-1].replace(self.suffix,""))

    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]
        #print("Doing {}".format(self.dataCPU['filename'][index]))
        return self.dataCPU['image'][index], self.dataCPU['label'][index], self.dataCPU['filename'][index]


class ROSE(data.Dataset):
    def __init__(self, listpath, folderpaths, is_training=False, is_testing=False, constcorner=False):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.indices = [] 
        self.to_tensor = transforms.ToTensor()
        self.crop_size = 128
        self.is_training = is_training
        self.is_testing = is_testing
        self.constcorner = constcorner

        self.loadCPU()

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist):
            components = entry.split('.')
            filename = components[0]

            im_path = pjoin(self.imgfolder, filename) + '.tif'
            gt_path = pjoin(self.gtfolder, filename) + '.tif'

            img = Image.open(im_path)
            img = self.to_tensor(img)

            gt = np.expand_dims(np.array(Image.open(gt_path))/255., 0)
            gt = torch.from_numpy(gt)

            #normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval

            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['filename'].append(str(filename))

    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.is_training is True:
            # crop to patchsize. compute top-left corner first
            # have a hunch that dmt cannot handle completely black image. so retry if GT has no structure --- this is hoping network outputs something when GT is non-black. 
            # Actually, we cannot have multiple DMT trainings in parallel because they access the same dipha txt files
            C, H, W = torch_img.shape
            #while True:
            if self.constcorner:
                corner_h = 128
                corner_w = 128
            else:
                corner_h = np.random.randint(low=0, high=H-self.crop_size)
                corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

            #print(corner_h, corner_w)

                #if torch.sum(new_torch_gt)  != 0. :
                #    break
                #else:
                #    print("Black GT; Retrying patch generateion...")

        #torch_img = new_torch_img
        #torch_gt = new_torch_gt

        return torch_img, torch_gt, self.dataCPU['filename'][index]




if __name__ == "__main__":
    flag = "training"
    
    dst = CREMI('data-lists/CREMI/validation-list.csv', ['data/CREMI/train-volume.tif', 'data/CREMI/train-labels.tif'])
    # dst = ISBI2013('data-lists/ISBI2013/train-list.csv', ['data/ISBI2013/train-volume.tif', 'data/ISBI2013/train-labels.tif'], is_training= True)

    # dst = DRIVE('data-lists/DRIVE/train-list.csv', ['data/DRIVE/images', 'data/DRIVE/1st_manual'], is_training= True)
 
    validationloader = data.DataLoader(dst, shuffle=False, batch_size=1, num_workers=1)

    ## dataloader check
    # import pdb; pdb.set_trace()
    batch = next(iter(validationloader))
    input, target = batch
    # import pdb; pdb.set_trace()
