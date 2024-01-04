import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import *
from PIL import Image
from data_loader import PascalVOC2012
from testset import test_dataset
from model_seg_flow import FeaturePath
from model_seg_flow import FlowSegPath
from model_seg_flow import net_utils
from model_seg_flow import Net
from model_seg_flow.settings import settings
import cv2
ap = '/home/zhangwei/spatialaffinitynetwork/VOCdevkit/VOC2012/PNGImage/'
pic = os.listdir(ap)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.cuda.set_device(3)
torch.cuda.empty_cache()#清除PyTorch缓存。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net.OpticalSegFLow(device=device, num_channels=1, num_levels=6, use_cost_volume=True)#创建一个基于光流法和U-Net的肺部CT图像分割模型，其中OpticalSegFLow是一个自定义的类，实现了这个模型。
net.to(device)#将模型移动到GPU上进行训练或推理。
model_save_path1 = '/home/zhangwei/spatialaffinitynetwork/model.pth1'
#checkpoint = torch.load(model_save_path)
#checkpoint = torch.load(model_save_path1,  map_location='cuda:0')
#epochs = checkpoint['epoch']
#net.load_state_dict(checkpoint['seg_flow_net'])
final_dice = []
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
class Solver(object):
    DEFAULTS = {}
    def __init__(self, data_loader, data_loader_testset, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader
        self.data_loader_testset = data_loader_testset
        self.forward_imgs=[]
        self.backward_imgs=[]
        self.a = []
        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()
    def build_model(self):
        # Define a generator and a discriminator
        self.FCN8 = FCN8s(n_class=self.num_classes)
        print('gggg',self.num_classes)
        self.guidance_module = VGG16Modified(n_classes=2)
        # Optimizers
        self.optimizer = torch.optim.Adam(self.guidance_module.parameters(), self.lr)
        # Print networks
        self.print_network(self.guidance_module, 'Guidance Network')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = VGG16Modified(n_classes=2)
        ##torch.cuda.set_device(2)
        self.FCN8.to(device)
        self.G.to(device)
        model_data = torch.load(self.fcn_model_path)
        self.FCN8.load_state_dict(model_data)
        self.FCN8.eval()
        self.guidance_module.copy_params_from_vgg16(self.vgg_model_path)
        self.guidance_module.to(device)
    def mask_dice(self,prediction, groundtruth):
        intersection = torch.logical_and(groundtruth, prediction)
        union = torch.logical_or(groundtruth, prediction)
        if torch.sum(union) + torch.sum(intersection) == 0:
            return 1
        dice_score = 2*torch.sum(intersection) / (torch.sum(union) + torch.sum(intersection))
        return dice_score.item()
    def pack_tensor(self,image, is_mask=False):
        path = ap + image
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = image / 255
        image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)
        return image
    def get_data(self,pics):
        for i in range(len(pics)):
            if len(pics)%2 == 0 :
                mid_img = 'slice-'+ str(len(pics)//2)+'-ori.png'
                if i < len(pics)//2 :
                    if i+1 == len(pics)//2:
                        forward_img = ('slice-'+ str(i)+'.png',mid_img)
                    else:
                        forward_img = ('slice-'+ str(i)+'.png','slice-'+ str(i+1)+'.png')
                    self.forward_imgs.append(forward_img)
                if i > len(pics)//2 :
                    if i-1 == len(pics)//2:
                        backward_img = ('slice-'+ str(i)+'.png',mid_img)
                    else:
                        backward_img = ('slice-'+ str(i)+'.png','slice-'+ str(i-1)+'.png')
                    self.backward_imgs.append(backward_img)
            elif len(pics)%2 != 0 :
                mid_img = 'slice-'+ str(len(pics)//2-0.5)+'.png'
        return self.forward_imgs, self.backward_imgs
    def get_warp(self,imgs):
        forward = []
        with torch.no_grad():
            for i in range(len(imgs)):
                img1,img2 = imgs[i]
                image1 = self.pack_tensor(img1, is_mask=False) .to(device)
                image2 = self.pack_tensor(img2, is_mask=False) .to(device)
                image1 = image1.unsqueeze(0).to(device)
                image2 = image2.unsqueeze(0).to(device)
                #print(image1.shape)
                flows_f, segs_f, fp1, fp2, flows_b, segs_b = net(image2, image1, training=False)
                warp = net_utils.flow_to_warp(flows_f[0]) 
                forward.append(warp)
        return forward
    def train(self):
        # The number of iterations per epoch
        print("start training")
        self.guidance_module.to(device)
        iters_per_epoch = len(self.data_loader)
        print('iters_per_epoch = len(self.data_loader)',len(self.data_loader))
        fixed_x = [] 
        fixed_target = [] 
        fixed_trmask = [] 
        for i, (images, target,tr_mask,tr_images,images_1,num) in enumerate(self.data_loader):
            print('images',images)
            print('target',torch.sum(target))
            print('tr_mask',tr_mask)
            print('tr_images',tr_images)
            print('images_1',images_1)
            id_sum =num.tolist()
            print('type',type(id_sum))
            fixed_x.append(images) 
            fixed_target.append(target) 
            fixed_trmask.append(tr_mask)
            if i == 1:
                break
        print("sample data")
        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0).to(device)
        fixed_x = self.to_var(fixed_x, volatile=True).to(device)
        fixed_target = torch.cat(fixed_target, dim=0).to(device)
        fixed_trmask = torch.cat(fixed_trmask, dim=0).to(device)
        # lr cache for decaying
        lr = self.lr
        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0
        # Start training
        start_time = time.time()
        criterion = CrossEntropyLoss2d(size_average=False, ignore_index=-1).to(device)
        #criterion.to(device)
        #self.get_data(pic)
        #self.forward_warps = self.get_warp(self.forward_imgs)
        #self.backward_warps = self.get_warp(self.backward_imgs)
        #print(len(self.forward_warps),len(self.backward_warps))
        dice_set = []
        for e in range(start, self.num_epochs):
            for i, (images, target, tr_mask,tr_images,images_1,num) in enumerate(self.data_loader):
                print('**********************************************************',i)
                N = images.size(0)
                # Convert tensor to variable
                images = self.to_var(images)
                num = num.detach().cpu().numpy().astype(int)
                images = images.repeat(1,3,1,1)
                images_1 = self.to_var(images_1)
                #print(coarse_map.shape)
                coarse_map = self.to_var(target)
                coarse_map_ = (coarse_map + 1) * -1
                coarse_map = torch.cat([coarse_map_, coarse_map], dim=1)
                print(f'images:{images.shape}')
                
                print(f'coarse_map:{coarse_map.shape}')
                
                refined_map = self.guidance_module(images.to(device), coarse_map.to(device)).to(device)
                
                tr_mask = self.to_var(tr_mask)
                tr_mask = torch.squeeze(tr_mask, dim=1).to(device)

                print("tr_mask",tr_mask.shape)
                print("coarse_map",coarse_map.shape)


                images_1 = images_1.unsqueeze(0).to(device)
                tr_images = tr_images.to(device)
                images_1 = images_1.to(device)
                tr_images = tr_images.repeat(1,3,1,1)
                tr_mask = torch.squeeze(tr_mask, dim=1).to(device)
                refined_map_ = refined_map.clone()
                refined_map_ = refined_map*255
                refined_map_[refined_map_>=127.5] = 255
                refined_map_[refined_map_< 127.5] = 0
                refined_map_ = refined_map_.detach().cpu().numpy()
                refined_map_ = refined_map_/255
                refined_map_ = torch.from_numpy(refined_map_)

                print("refined_map_",refined_map_.shape)

                images = images.detach().cpu().numpy()
                tr_mask_= tr_mask.clone()
                tr_mask_ =tr_mask_ .detach().cpu().numpy()
                dice = self.mask_dice(refined_map_[0,1].to(device), tr_mask[0].to(device).long())
                #dice_set.append(dice)
                print('dice',dice)
               # print('ffffffff',output_image1.size)
                #output_image1 = torch.from_numpy(output_image1)
                #refined_map = self.guidance_module(images.to(device), coarse_map.to(device)).to(device)
                refined_map_ = refined_map.clone()
                refined_map_ = refined_map_*255
                refined_map_[refined_map_>=127.5] = 255
                refined_map_[refined_map_< 127.5] = 0
                refined_map_ = refined_map_.detach().cpu().numpy()
                #refined_map_  = refined_map_.squeeze(0)
                #channel_1 = refined_map_[1]
                #output_image = np.zeros((128, 128,1), dtype=np.uint8)
                #output_image[:, :, 0] = channel_1
                #print('refine',refined_map_[0].shape)
                cv2.imwrite('/home/zhangwei/spatialaffinitynetwork/spatial_affinity/refined_map/refined_map-'+str(i)+'.png', refined_map_[0,1])
                # # Compute classification accuracy of the discriminator
                # if (i+1) % self.log_step == 0:
                #     accuracies = self.compute_accuracy(real_feature, real_label, self.dataset)
                #     .numpy()]
                #     if self.dataset == 'CelebA':
                #         print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                #     else:
                #         print('Classification Acc (8 emotional expressions): ', end='')
                #     print(log)
                # Logging
                #loss = {}
                #loss['loss'] = softmax_ce_loss.item() 
                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)
                    softmax_ce_loss = (criterion(refined_map.to(device), tr_mask.to(device).long()) * 0.0001 / N) 
                    print('dddddddddddddddd',type(softmax_ce_loss),softmax_ce_loss)
                    print('softmax_ce_loss',softmax_ce_loss)
                    #softmax_ce_loss = torch.tensor(softmax_ce_loss,requires_grad=True)
                    softmax_ce_loss.to(device)
                    self.reset_grad()
                    softmax_ce_loss.backward() 
                    self.optimizer.step()
                    loss = {}
                    loss['loss'] = softmax_ce_loss.item() 
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.guidance_module.state_dict(),
                        os.path.join(self.model_save_path, 'spatial4.pth'))
                    print('model saved')
                    if int(e+1)% 5 == 0 :
                        self.test()
            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                self.update_lr(lr)
                print ('Decay learning rate to lr: {}.'.format(lr))           
    def labels_to_rgb(self,labels):
        return
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))
    def load_pretrained_model(self):
        self.guidance_module.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_spatial4.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    def reset_grad(self):
        self.optimizer.zero_grad()
    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)
    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x
    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        elif dataset == 'Flowers':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy
    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
    def test(self):
        G_path = os.path.join('/home/zhangwei/spatialaffinitynetwork/spatial_affinity/checkpoint/spatial4.pth')
        self.G.load_state_dict(torch.load(G_path, map_location='cuda:0'))
        self.G.eval()
        for i, (test_images, test_target, op_name) in enumerate(self.data_loader_testset):
            test_images = self.to_var(test_images)
            test_images = test_images.repeat(1,3,1,1)
            coarse_target = self.to_var(test_target)
            coarse_target_ = (coarse_target + 1) * -1
            coarse_target = torch.cat([coarse_target_, coarse_target], dim=1)
            refined_coarse_map = self.G(test_images.to(device),coarse_target.to(device))
            refined_coarse_map_ = refined_coarse_map.clone()
            refined_coarse_map_ = refined_coarse_map_*255
            refined_coarse_map_[refined_coarse_map_>=127.5] = 255
            refined_coarse_map_[refined_coarse_map_< 127.5] = 0
            refined_coarse_map_ = refined_coarse_map_.detach().cpu().numpy()
            #op_loc = str(op_name).split('/')[08
            #op_loc_ = str(op_name).split('/')[10].split(',')[0].split('.')[0]
            cv2.imwrite('/home/zhangwei/optical_flow/result/plain-optical-flow/result-imagespleen/LIDC-IDRI-0021/nodule-0/mask-sp/slice-'+str(i)+'.png', refined_coarse_map_[0,1])
            print('/home/zhangwei/optical_flow/result/plain-optical-flow/result-imagespleen/LIDC-IDRI-0001/nodule-0/mask-sp/slice-'+str(i)+'.png')
            print ('test',i)
        def my_sort( strs):
                return int(strs.split('.')[0].split('-')[1])
        path ='/home/zhangwei/optical_flow/result/plain-optical-flow/result-imagespleen/LIDC-IDRI-0021/nodule-0/mask-sp/'
        path1 = '/home/zhangwei/optical_flow/data/LIDC-IDRI-IDspleen/LIDC-IDRI-0021/nodule-0/mask-0/'
        #path_op = '/home/zhangwei/optical_flow/result/plain-seg-flow/result-image3d/LIDC-IDRI-0009/nodule-0/mask-0//'
        images0 = os.listdir('/home/zhangwei/optical_flow/data/LIDC-IDRI-IDspleen/LIDC-IDRI-0021/nodule-0/mask-0/')
        masks0 = os.listdir('/home/zhangwei/optical_flow/result/plain-optical-flow/result-imagespleen/LIDC-IDRI-0021/nodule-0/mask-sp/')
        images0.sort(key = my_sort)
        masks0.sort(key = my_sort)
        dice_2=[]
        for n in range(len(images0)):
            print('ddddddddd')
            path_im = path1 +str(images0[n])
            path_imask = path + str(masks0[n])
            #print(path_im)0
            #print(path_imask)
            fn_im = cv2.imread(path_im)
            fn_op = cv2.imread(path_imask)
            fn_mask = cv2.imread(path_imask)
            def mask_dice(prediction, groundtruth):
                prediction = torch.from_numpy(prediction)
                groundtruth = torch.from_numpy(groundtruth)
                intersection = torch.logical_and(groundtruth, prediction)
                union = torch.logical_or(groundtruth, prediction)
                if torch.sum(union) + torch.sum(intersection) == 0:
                    return 1
                dice_score = 2*torch.sum(intersection) / (torch.sum(union) + torch.sum(intersection))
                return dice_score.item() 
            fn_im[fn_im <= 127.5] = 0
            fn_im[fn_im >= 127.5] = 1
            fn_op[fn_op == 1] = 0
            fn_op[fn_op >= 127.5] = 1
            fn_mask[fn_mask >= 127.5] = 1
            print(np.min(fn_im))
            print(np.min(fn_op))
            print(np.min(fn_mask))
            dice_1 = mask_dice(fn_op,fn_im)
            print(path_im)
            print(path_imask)
            print(dice_1)
            #print('fn_op, ', np.min(fn_op),np.max(fn_op))
            #print('fn_mask, ', np.min(fn_mask),np.max(fn_mask))
            #print('fn_im, ', np.min(fn_im),np.max(fn_im))
            #print('Dice', dice_1)
            #if dice_1 == 0.0:
                #print(path_im)
                #print(path_imask)
                #print('Dice', dice_1)
                #print('fn_im, ', np.min(fn_im),np.max(fn_im))
            dice_2.append(dice_1)
            #print('noid',np.mean(dice_2))
        print(len(dice_2))
        print(np.mean(dice_2))
        if np.mean(dice_2)>0.98:
            os._exit()
        final_dice.append(np.mean(dice_2))
        print('test_dice',final_dice)
        print('best_dice',np.max(final_dice))
