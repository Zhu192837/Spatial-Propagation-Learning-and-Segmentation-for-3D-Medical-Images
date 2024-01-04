import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import cv2
class RandomCropGenerator(object):
    def __call__(self, img):
        self.x1 = random.uniform(0, 1)
        self.y1 = random.uniform(0, 1)
        return img

class RandomCrop(object):
    def __init__(self, size, padding=0, gen=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self._gen = gen

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        if self._gen is not None:
            x1 = math.floor(self._gen.x1 * (w - tw))
            y1 = math.floor(self._gen.y1 * (h - th))
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

        return img.crop((x1, y1, x1 + tw, y1 + th))

##################################################################
class test_dataset(Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    '''
    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
    12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    '''
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
               128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
               64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

    def my_sort(self,strs):
        return int(strs.split('.')[0].split('-')[1])
    def __init__(self, root, transform, crop_size, image_size, mode='train'):
        self.root = '/home/zhangwei/spatialaffinitynetwork/'

        if mode == 'tri':
            self.split = 'tri'
        else:
            self.split = 'img'

        self.crop_size = crop_size
        self.image_size=image_size

        self._transform = transform
        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette.append(0)
        # VOC2011 and others are subset of VOC2012
        dataset_dir = '/home/zhangwei/optical_flow/data/LIDC-IDRI-IDliver/LIDC-IDRI-0001/nodule-0/images/'
        dataset_dir1 = '/home/zhangwei/optical_flow/result/plain-seg-flow/result-image3d/LIDC-IDRI-0001/nodule-0/mask-0/'
        self.files = collections.defaultdict(list)
        self.files_1 = collections.defaultdict(list)
        for split in ['tri','img']:
            imgsets_file =  os.listdir(dataset_dir1)
            imgsets_file.sort(key = self.my_sort)
            #print(imgsets_file)
            for did in imgsets_file:
                #did = did.strip()
                img_file = os.path.join(dataset_dir, did)
                lbl_file = os.path.join(dataset_dir1,did)
                #trimg_file = os.path.join(dataset_dir, 'PNGImage/%s' % did)
                #trmask_file = os.path.join(dataset_dir, 'Segmentations/%s' % did)
                #print(lbl_file)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })
        print( self.files)
    def __len__(self):
        return len(self.files[self.split])
    def colorize_mask(self,mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.palette)
    def colorize_mask_batch(self,masks):
        color_masks = np.zeros((masks.shape[0],3,masks.shape[1],masks.shape[2]))
        toTensor = transforms.ToTensor()
        for i in range(masks.shape[0]):
            color_masks[i] = np.array(self.colorize_mask(masks[i])).transpose(2,0,1)
        return torch.from_numpy(color_masks).float()
    def process_mask(self, mask):
        mask[mask>=127.5] = 255
        mask[mask<127.5] = 0
        mask = mask
        return mask
    def pack_tensor(self, path, is_mask=False):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if is_mask:
            image = self.process_mask(image)
        image = image / 255
        image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)
        return image
    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        img_file = data_file['img']
        lbl_file = data_file['lbl']
        return  self.transform(img_file, lbl_file)
    def transform(self,img_file,lbl_file):
        img = self.pack_tensor(img_file, is_mask=False)
        lbl = self.pack_tensor(lbl_file, is_mask=True)
        name = str(img_file)
        return img, lbl,name
    def untransform(self, img):
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1] / 255
        img = img.transpose(2, 0, 1)
        return img
    def untransform_batch(self, img):
        img = img.cpu().numpy()
        for i in range(img.shape[0]):
            img[i] = self.untransform(img[i])
        return img
def get_loader_test(image_path, crop_size, image_size, batch_size, transform=False, dataset='test_dataset', mode='tri'):
    """Build and return data loader."""
    print()
    if dataset == 'test_dataset':
        dataset = test_dataset(image_path, transform, crop_size, image_size, mode)
    print(dataset)
    shuffle = False
    if mode == 'tri':
        shuffle = True
    data_loader_testset = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader_testset