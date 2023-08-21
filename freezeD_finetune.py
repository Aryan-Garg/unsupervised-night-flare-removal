#!/usr/bin/env python

# All Imports

### Python Imports
import sys
import numpy as np
from time import sleep
import os
import argparse
import random
import skimage
import cv2
from tqdm import tqdm


### Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


### Torchvision
import torchvision
from torchvision import utils as vutils

### skimage & Pytorch Guided Filter
from guided_filter_pytorch.guided_filter import GuidedFilter
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

### WandB, Pytorch Lightning & torchsummary
import wandb

import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import DeviceStatsMonitor, TQDMProgressBar, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from torchsummary import summary

### Custom Modules
import load_data as DA
from Net import *

seed_everything(42)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data & Chekpoint Paths:
    parser.add_argument("--out_dir", type=str,  default='./results/oxford_mini_LOL/', help="Location at which to save the light-effects suppression results.")
    parser.add_argument("--data_dir", type=str, default='./datasets/oxford_mini/', help="Directory containing images with light-effects for demo")
    parser.add_argument("--load_model", type=str, default=None, help="model to initialize with")
    
    # Image Loading Parameters:
    parser.add_argument("--load_size", type=str, default="Resize", help="Width and height to resize training and testing frames. None for no resizing, only [512, 512] for no resizing")
    parser.add_argument("--crop_size", type=str, default="[512, 512]", help="Width and height to crop training and testing frames. Must be a multiple of 16")
    parser.add_argument('--use_gray', action='store_true')

    # Training Parameters:
    # epochs = ( All samples / batch_size ) * iterations
    parser.add_argument("--iters", type=int, default=60, help="iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the model.")
    parser.add_argument("--epochs", type=int, default=5, help="No of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")

    # Device Parameters:
    parser.add_argument('--gpu_id', type=int, default=0)

    # Modes: train, test, fine-tune
    parser.add_argument('--mode', type=str, default="test", help='train, test, finetune')
    
    # Logging
    parser.add_argument('--test_log_img_freq', type=int, default=1, help='Frequency of logging test images during training')
    parser.add_argument('--train_log_img_freq', type=int, default=1, help='Frequency of logging train images during training')
    parser.add_argument('--experiment_name', '-en', type=str, default='default', help='name of the experiment')
    return parser.parse_args()


def get_LFHF(image, rad_list=[4, 8, 16, 32], eps_list=[0.001, 0.0001]):
    def decomposition(guide, inp, rad_list, eps_list):
        LF_list = []
        HF_list = []
        for radius in rad_list:
            for eps in eps_list:
                gf = GuidedFilter(radius, eps)
                LF = gf(guide, inp)
                LF[LF>1] = 1 
                LF_list.append(LF)
                HF_list.append(inp - LF)
        LF = torch.cat(LF_list, dim=1)
        HF = torch.cat(HF_list, dim=1)
        return LF, HF
    image = torch.clamp(image, min=0.0, max=1.0)
    # Compute the LF-HF features of the image
    img_lf, img_hf = decomposition(guide=image, 
                                   inp=image, 
                                   rad_list=rad_list,
                                   eps_list=eps_list)
    return img_lf, img_hf


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class Vgg16ExDark(torch.nn.Module):
    def __init__(self, load_model=None, requires_grad=False):
        super(Vgg16ExDark, self).__init__()
        # Create the model
        self.vgg_pretrained_features = visionmodels.vgg16(pretrained=True).features
        if load_model is None:
            print('Vgg16ExDark needs a pre-trained checkpoint!')
            raise Exception
        else:
            print('Vgg16ExDark initialized with %s'% load_model)
            model_state_dict = torch.load(load_model)
            model_dict       = self.vgg_pretrained_features.state_dict()
            model_state_dict = {k[16:]: v for k, v in model_state_dict.items() if k[16:] in model_dict}
            self.vgg_pretrained_features.load_state_dict(model_state_dict)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [3, 8, 15, 22] 
        out = []
        for i in range(indices[-1]+1):
            X = self.vgg_pretrained_features[i](X)
            if i in indices:
                out.append(X)
        return out


class PerceptualLossVgg16ExDark(nn.Module):
    def __init__(self, vgg=None, 
                 load_model=None,
                 weights=None, 
                 indices=None, 
                 normalize=True):
        super(PerceptualLossVgg16ExDark, self).__init__()        
        if vgg is None:
            self.vgg = Vgg16ExDark(load_model)
        else:
            self.vgg = vgg
        self.vgg     = self.vgg.cuda()
        self.criter  = nn.L1Loss()
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]
        self.indices = indices or [3, 8, 15, 22]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], 
                                       [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criter(x_vgg[i], y_vgg[i].detach())
        return loss


class StdLoss(nn.Module):
    def __init__(self):
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


class ExclusionLoss(nn.Module):
    def __init__(self, level=3):
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


def gradient(pred):
    D_dy      = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx      = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        
    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)


def smooth_loss(pred_map):
    dx, dy   = gradient(pred_map)
    dx2, dxdy= gradient(dx)
    dydx, dy2= gradient(dy)
    loss     =  (dx2.abs().mean()  + dxdy.abs().mean()+ 
                 dydx.abs().mean() + dy2.abs().mean())
    return loss


def rgb2gray(rgb):
    gray = 0.2989*rgb[:,:,0:1,:] + \
    	   0.5870*rgb[:,:,1:2,:] + \
    	   0.1140*rgb[:,:,2:3,:]
    return gray


def validate(dle_net, 
             inputs):
    print('Validation not possible since there are no labels!')
    raise Exception


def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)


def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


def calc_psnr_masked(im1, im2, mask):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y[mask], im2_y[mask])


def calc_ssim_masked(im1, im2, mask):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y[mask], im2_y[mask])


class LIT_NightEnhancement(pl.LightningModule):
    
    def __init__(self, dle_net, args):
        super().__init__()
        self.save_hyperparameters(ignore=['dle_net'])
        
        self.mode = args.mode # train/test/finetune
        self.model = dle_net.to(device_0)

        if self.mode == 'finetune':
            print(f"For finetuning:\n")
            # freeze all layers except the last one
            print(self.model)
            for name, param in self.model.named_parameters():
                print(name, param.shape)
                if name != 'conv10.weight' and name != 'conv10.bias':
                    param.requires_grad = False
            
        self.learning_rate = args.learning_rate
        self.training_step_outputs = []
        self.testing_step_outputs = []

        self.test_log_img_freq = args.test_log_img_freq
        self.train_log_img_freq = args.train_log_img_freq


    def configure_optimizers(self):
        optimizer_dle_net = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        # return [optimizers] , [schedulers]
        return [optimizer_dle_net], []


    def forward(self, x):
        return self.model(x)
        

    def get_reconstruction_loss(self, x, y):
        return 1.* F.l1_loss(x, y)    


    def get_cc_loss(self, dle_pred): # color constancy loss
        dle_pred_cc = torch.mean(dle_pred, dim=1, keepdims=True)
        cc_loss = (F.l1_loss(dle_pred[:, 0:1, :, :], dle_pred_cc) + \
                   F.l1_loss(dle_pred[:, 1:2, :, :], dle_pred_cc) + \
                   F.l1_loss(dle_pred[:, 2:3, :, :], dle_pred_cc))*(1/3) 
        return 1.*cc_loss


    def get_smooth_loss(self, le_pred):
        return 1.*smooth_loss(le_pred)


    def get_excl_loss(self, dle_pred, le_pred):
        excl_loss = ExclusionLoss().type(torch.cuda.FloatTensor)                      
        loss = excl_loss(dle_pred, le_pred)
        return 0.01 * loss


    def training_step(self, batch, batch_idx):
        img_in = batch['img_in']
        
        le_pred = self.model(img_in)
        dle_pred= img_in + le_pred

        reconstruction_loss = self.get_reconstruction_loss(dle_pred, img_in) # pred , target ALWAYS!
        cc_loss = self.get_cc_loss(dle_pred)
        smooth_loss = self.get_smooth_loss(le_pred)
        excl_loss = self.get_excl_loss(dle_pred, le_pred)

        loss = reconstruction_loss + cc_loss + excl_loss + smooth_loss
        
        self.log("recontruction_loss", reconstruction_loss, prog_bar=True, logger=True)
        self.log("cc_loss", cc_loss, prog_bar=True, logger=True)
        self.log("excl_loss", excl_loss, prog_bar=True, logger=True)
        self.log("smooth_loss", smooth_loss, prog_bar=True, logger=True)

        self.log("total_loss", loss, prog_bar=True, logger=True)

        self.training_step_outputs.append([img_in, le_pred, dle_pred])

        return loss


    def on_train_epoch_end(self):
        # save images to logger
        img_in, le_pred, dle_pred = self.training_step_outputs[-1]
        if self.current_epoch % self.train_log_img_freq == 0:
            img_in = img_in.cpu().detach().numpy().transpose(1,2,0)
            le_pred = le_pred.cpu().detach().numpy().transpose(1,2,0)
            dle_pred = dle_pred.cpu().detach().numpy().transpose(1,2,0)
            
            self.logger.experiment.add_image('img_in', img_in, self.current_epoch)
            self.logger.experiment.add_image('le_pred', le_pred, self.current_epoch)
            self.logger.experiment.add_image('dle_pred', dle_pred, self.current_epoch)
            

    def validation_step(self, batch, batch_idx):
        pass


    def test_step(self, batch, batch_idx):
        img_in = batch['img_in'].to(device_0)
        # print(img_in.shape, img_in.device, img_in.dtype, type(img_in))
        
        le_pred = self.model(img_in)
        dle_pred= img_in + le_pred

        reconstruction_loss = self.get_reconstruction_loss(dle_pred, img_in) # pred , target ALWAYS!
        cc_loss = self.get_cc_loss(dle_pred)
        smooth_loss = self.get_smooth_loss(le_pred)
        excl_loss = self.get_excl_loss(dle_pred, le_pred)

        testloss = reconstruction_loss + cc_loss + excl_loss + smooth_loss
        
        self.log("test_recontruction_loss", reconstruction_loss, prog_bar=True, logger=True)
        self.log("test_cc_loss", cc_loss, prog_bar=True, logger=True)
        self.log("test_excl_loss", excl_loss, prog_bar=True, logger=True)
        self.log("test_smooth_loss", smooth_loss, prog_bar=True, logger=True)

        self.log("test_total_loss", testloss, prog_bar=True, logger=True)

        self.testing_step_outputs.append([img_in[0], le_pred[0], dle_pred[0]])

        return testloss
    

    def on_test_epoch_end(self):
        if self.current_epoch % self.test_log_img_freq == 0:
            for img_in, le_pred, dle_pred in self.testing_step_outputs:
                img_in = img_in.cpu().detach().numpy().transpose(1,2,0)
                le_pred = le_pred.cpu().detach().numpy().transpose(1,2,0)
                dle_pred = dle_pred.cpu().detach().numpy().transpose(1,2,0)
            
                self.logger.log_image(key='Test Outputs', 
                                      images=[img_in, le_pred, dle_pred], 
                                      caption=["Original", "Light Effects", "Model Output"])


if __name__ == '__main__':
    args = get_args()

    isFinetune = False
    if args.mode == 'finetune':
        isFinetune = True

    args.imgin_dir = args.data_dir

    device_0 = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    args.imgs_dir = args.out_dir
    if not os.path.exists(args.imgs_dir):
        os.makedirs(args.imgs_dir)

    if args.use_gray:
        channels = 1
    else:
        channels = 3
        
    dle_net = Net(input_nc=channels, output_nc=channels)
    # dle_net = nn.DataParallel(dle_net).cuda()

    if args.load_model is not None:
        dle_net_ckpt_file = args.load_model
        loaded_ckpt = torch.load(dle_net_ckpt_file, map_location=device_0)
        # print(loaded_ckpt.keys())
        dle_net.load_state_dict(loaded_ckpt, strict=False)

    # summary(dle_net.to(device_0), (3,512,512))
    nightGAN = LIT_NightEnhancement(dle_net, args)

    da_list  = sorted([(args.imgin_dir+ file) for file in os.listdir(args.imgin_dir)])
    demo_list   = da_list

    Dele_Loader  = DataLoader(DA.loadImgs(args, demo_list, mode='demo'),
                                batch_size  = args.batch_size, 
                                shuffle     = True, 
                                num_workers = 16, 
                                drop_last   = False)

    # for i, batch in enumerate(Dele_Loader):
    #     print(i, batch['img_in'].shape)
    #     exit(0)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="loss")
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
    # early_stop_callback = EarlyStopping(monitor="loss", patience=99)

    wandb.login()
    
    wandb_logger = WandbLogger(project='night_restoration',
                           name=f'{args.experiment_name}',
                           config=vars(args),
                           job_type='finetuning',
                           log_model="all")
    # call trainer
    trainer = Trainer(fast_dev_run=False,
                      inference_mode=isFinetune, 
                      min_epochs=args.epochs,
                      max_epochs=args.epochs,
                      devices=1,
                      accelerator="gpu",
                      precision="32", # Mixed precision ---> invalid for testing
                      deterministic=True,
                      enable_checkpointing=True,
                      callbacks=[checkpoint_callback, lr_monitor],
                      gradient_clip_val=None, # TODO: Need to run an experiment to find the best value
                      log_every_n_steps=50,
                      logger=wandb_logger, # The absolute best: wandb <3
                      enable_progress_bar=True,
                      )

    # fit model
    if args.mode == 'train' or args.mode == 'finetune':
        trainer.fit(nightGAN, Dele_Loader)
    
    else:
        # trainer.test(nightGAN, Dele_Loader)
        test_model = dle_net.to(device_0)

        with torch.inference_mode():
            for i, batch in tqdm(enumerate(Dele_Loader)):
                img_in = batch['img_in'].to(device_0)
                le_pred = nightGAN.model(img_in)
                dle_pred = img_in + le_pred
                
                orig = img_in[0].cpu().detach()
                light_effects_pred = le_pred[0].cpu().detach()
                model_pred = dle_pred[0].cpu().detach()

                all_save = torch.cat([orig, light_effects_pred, model_pred], dim=2)

                vutils.save_image(orig, f'{args.imgs_dir}/orig_{i}.png')
                vutils.save_image(light_effects_pred, f'{args.imgs_dir}/light_effects_pred_{i}.png')
                vutils.save_image(model_pred, f'{args.imgs_dir}/model_pred_{i}.png')
                vutils.save_image(all_save, f'{args.imgs_dir}/all_{i}.png')

    
    wandb.finish()