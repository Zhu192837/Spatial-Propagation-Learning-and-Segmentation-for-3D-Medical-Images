import torch
import numpy as np
import cv2
import os
from torch.utils.data.dataset import Dataset
try:
    from sklearn.externals import joblib
except:
    import joblib

class LIDC_IDRI_flow(Dataset):

    def __init__(self, root=None, load=False, training=True):
        self.ap = os.getcwd()
        if root is None:
            root = os.path.join(self.ap, 'data/LIDC-IDRI-slices/')
        print("Loading file ", root)
        print('This data is for flow ' + ('training' if training else 'test'))
        self.root = root
        self.load = load
        self.training = training
        
        if self.load:
            print('load pkl')
            self.train = joblib.load(os.path.join(self.ap, 'save/immediate_data/train_flow.pkl')) 
            self.test = joblib.load(os.path.join(self.ap, 'save/immediate_data/test_flow.pkl')) 
            print('[INFO]The length of train: ', len(self.train))
            print('[INFO]The nodule of test: ', len(self.test))
        else:
            print('split dataset')
            self.split_dataset()
    
    def split_dataset(self):
        root = self.root
        self.data = dict() # { noduleId: path }
        files = os.listdir(root)
        nodule_id = 0
        for file in files:
            nodules = os.listdir(os.path.join(root, file))
            for nodule in nodules:
                self.data[nodule_id] = os.path.join(root, file, nodule)
                nodule_id += 1

        nodule_ids = list(range(len(self.data)))
        train_num = int(len(self.data) * 0.7)
        test_num = len(self.data) - train_num
        np.random.shuffle(nodule_ids)
        train_ids = np.random.choice(nodule_ids, size=train_num, replace=False)
        test_ids = list(set(nodule_ids) - set(train_ids))
        assert len(test_ids) == test_num
        self.train = [] # [((img1, img2), (path, None)), ((img2, img3), (None, None))]
        self.test = [] # [(((img1, img2, ..., img-/2), (mask1, mask2, ...)), (...))]
        test_set = [] # [ path1, path2, path3, ... ]
        train_set = [] # [ path1, path2, path3, ... ]
        for i in nodule_ids:
            path = self.data[i]
            images = os.listdir(os.path.join(path, 'images'))
            masks = os.listdir(os.path.join(path, 'mask-0'))
            images.sort(key = self.my_sort)
            masks.sort(key = self.my_sort)
            if len(images) < 2:
                continue
            if i in train_ids:
                #self.train += self.data[i]
                train_set.append(path.replace('\\', '/'))
                mask_idx = [len(images) // 2]
                for j in range(len(images) - 1):
                    path1 = os.path.join(path, 'images', 'slice-' + str(j) + '.png')
                    path2 = os.path.join(path, 'images', 'slice-' + str(j + 1) + '.png')
                    mask_flag1 = os.path.join(path, 'mask-0', 
                                              'slice-' + str(j) + '.png') if j in mask_idx else None
                    mask_flag2 = os.path.join(path, 'mask-0',
                                              'slice-' + str(j + 1) + '.png') if (j + 1) in mask_idx else None
                    self.train += [
                            ((path1, path2), (mask_flag1, mask_flag2)),
                            ((path2, path1), (mask_flag2, mask_flag1))
                        ]
            elif i in test_ids:
                #self.test += self.data[i]
                test_set.append(path.replace('\\', '/'))
                init_idx = len(images) // 2
                backward_images = tuple(
                        [os.path.join(path, 'images', image) for image in images[init_idx::-1]]
                    )
                backward_masks = tuple(
                        [os.path.join(path, 'mask-0', mask) for mask in masks[init_idx::-1]]
                    )
                forward_images = tuple(
                        [os.path.join(path, 'images', image) for image in images[init_idx:]]
                    )
                forward_masks = tuple(
                        [os.path.join(path, 'mask-0', mask) for mask in masks[init_idx:]]
                    )
                self.test.append(
                        ((backward_images, backward_masks), (forward_images, forward_masks))
                    )
                
        joblib.dump(self.train, os.path.join(self.ap, 'save/immediate_data/train_flow.pkl')) 
        joblib.dump(self.test, os.path.join(self.ap, 'save/immediate_data/test_flow.pkl')) 
        joblib.dump(test_set, os.path.join(self.ap, 'save/immediate_data/test_set.pkl')) 
        joblib.dump(train_set, os.path.join(self.ap, 'save/immediate_data/train_set.pkl')) 
                
        del self.data
        print('[INFO]The length of train: ', len(self.train))
        print('[INFO]The nodule of test: ', len(self.test))
    
    def my_sort(self, strs):
        return int(strs.split('.')[0].split('-')[1])

    def process_mask(self, mask):
        mask[mask>=127.5] = 255
        mask[mask<127.5] = 0
        mask = mask
        return mask
    
    def pack_tensor(self, path, is_mask=False):
        path = path.replace('\\', '/').split('/')
        path = os.path.join(self.root, '/'.join(path[-4:]))
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if is_mask:
            image = self.process_mask(image)
        image = image / 255
        image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)
        return image

    def __getitem__(self, index):
        if self.training:
            images, mask_flags = self.train[index]
            image1 = self.pack_tensor(images[0], is_mask=False)
            image2 = self.pack_tensor(images[1], is_mask=False)
            mask1 = self.pack_tensor(mask_flags[0],
                                     is_mask=True) if mask_flags[0] is not None else torch.ones_like(image1) * -1
            mask2 = self.pack_tensor(mask_flags[1],
                                     is_mask=True) if mask_flags[1] is not None else torch.ones_like(image2) * -1
            return image1, image2, mask1, mask2
        else:
            backward, forward = self.test[index]
            backward_images = []
            backward_masks = []
            assert len(backward[0]) == len(backward[1])
            for i in range(len(backward[0])):
                backward_images.append(self.pack_tensor(backward[0][i], is_mask=False))
                backward_masks.append(self.pack_tensor(backward[1][i], is_mask=True))
            backward_images = torch.cat(backward_images, dim=0)
            backward_masks = torch.cat(backward_masks, dim=0)
            forward_images = []
            forward_masks = []
            assert len(forward[0]) == len(forward[1])
            for i in range(len(forward[0])):
                forward_images.append(self.pack_tensor(forward[0][i], is_mask=False))
                forward_masks.append(self.pack_tensor(forward[1][i], is_mask=True))
            forward_images = torch.cat(forward_images, dim=0)
            forward_masks = torch.cat(forward_masks, dim=0)
            
            assert backward_images.shape == backward_masks.shape
            assert forward_images.shape == forward_masks.shape
            
            return backward_images, backward_masks, forward_images, forward_masks

    # Override to give PyTorch size of dataset
    def __len__(self):
        if self.training:
            return len(self.train)
        else:
            return len(self.test)



# can't split datase, must provide train_set and test_set
class LIDC_IDRI_unet(Dataset):

    def __init__(self, root=None, load=None, training=True):
        assert load == True
        self.ap = os.getcwd()
        if root is None:
            root = os.path.join(self.ap, 'data/LIDC-IDRI-slices/')
        print("Loading file ", root)
        print('This data is for unet ' + ('training' if training else 'test'))
        self.root = root
        self.load = load
        self.training = training
        
        self.train = [] # [ (image, mask), (image, mask) ]
        self.test = [] # [ (image, mask), (image, mask) ]
        
        if self.load:
            print('load pkl')
            train_set = joblib.load(os.path.join(self.ap, 'save/immediate_data/train_set.pkl')) 
            test_set = joblib.load(os.path.join(self.ap, 'save/immediate_data/test_set.pkl')) 
            for nodule_path in train_set:
                LIDC_ID, nodule_id = nodule_path.split('/')[-2:]
                path = os.path.join(self.ap, 'data', 'LIDC-IDRI-slices', LIDC_ID, nodule_id)
                images = os.listdir(os.path.join(path, 'images'))
                self.train += [
                        (os.path.join(path, 'images', slices), 
                         os.path.join(path, 'mask-0', slices))
                        for slices in images
                    ]
            self.test = test_set
                
            print('[INFO]The nodule of train: ', len(train_set))
            print('[INFO]The nodule of test: ', len(test_set))
        else:
            raise NotImplementedError()

    def my_sort(self, strs):
        return int(strs.split('.')[0].split('-')[1])

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
        if self.training:
            image, mask = self.train[index]
            image = self.pack_tensor(image, is_mask=False)
            mask = self.pack_tensor(mask, is_mask=True)
            return image, mask
        else:
            images = []
            masks = []
            nodule_path = self.test[index]
            LIDC_ID, nodule_id = nodule_path.split('/')[-2:]
            nodule_path = os.path.join(self.ap, 'data', 'LIDC-IDRI-slices', LIDC_ID, nodule_id)
            slices = os.listdir(os.path.join(nodule_path, 'images'))
            slices.sort(key = self.my_sort)
            for sli in slices:
                t_path = os.path.join(nodule_path, 'images', sli)
                images.append(self.pack_tensor(t_path, is_mask=False))
                t_path = os.path.join(nodule_path, 'mask-0', sli)
                masks.append(self.pack_tensor(t_path, is_mask=True))
            images = torch.cat(images, dim=0)
            masks = torch.cat(masks, dim=0)
            return images, masks
            
            
    
    # Override to give PyTorch size of dataset
    def __len__(self):
        if self.training:
            return len(self.train)
        else:
            return len(self.test)





























