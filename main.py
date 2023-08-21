#!/usr/bin/env python

'''

    This file joins both networks:
    Light Effects Decomposition (LED) -> Light Effects Suppression + Low Light Enhancement Regions (LESLER)

    Description:
    You can train/test the whole paper with this file on the dataset of your choice.
        
    Arguments:


'''


# All Imports
### Python Imports
import sys
from typing import Any
import numpy as np
from time import sleep
import os
import argparse
import random
import skimage
import cv2
from tqdm.auto import tqdm
from itertools import chain


### Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

### Torchvision
import torchvision
from torchvision import transforms as T
from torchvision import utils as vutils
from torchvision.utils import make_grid


### skimage & Pytorch Guided Filter
from guided_filter_pytorch.guided_filter import GuidedFilter
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import keras_cv

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
from ENHANCENET import ENHANCENET
from Net import *
from networks import *
import argparse
from utils import *
from LED_initialized_Dataset import LED_init_Dataset 

seed_everything(42)


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

    def compute_gradient(self, img): # Just the difference between pixels
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


class LIT_LED(pl.LightningModule):
    '''
    Steps:
        1. Light effects and shading initialization loss
            Rationale: (image-decomposition paper: Gi and Li should be close to predicted G and L from Net)
        
        2. Compute Gradient Exclusion Loss --> Decorrelate Gi and J_init 
            Rationale: (image-decomposition paper: Decomposed layers should be as simple and uncorrelated as possible)
        
        3. Compute Color Constancy Loss
            Rationale: (Minimize color shifts in J_init)

        4. Compute Reconstruction Loss
            Rationale: (Ensure that the reconstructed image is close to the input image)

        5. Compute Smoothness Loss
            Rationale: (Ensure that the reconstructed Light Effects map (G) is smooth) 
            NOTE: Never mentioned in the paper but added in original repository!
    '''
    def __init__(self, model, args):
        super().__init__()

        self.save_hyperparameters(ignore=[model])
        
        self.model = model
        self.args = args

        # Weights for the losses
        self.lambda_recon       = 1.0
        self.lambda_excl        = 0.01
        self.lambda_smooth      = 1.0
        self.lambda_cc          = 1.0 

        # Initialize the loss(es)
        self.excl_loss          = ExclusionLoss().type(torch.cuda.FloatTensor)


    def configure_optimizers(self) -> Any:
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9,0.999))
        # Cosine Annealing Scheduler 
        # NOTE: For future improvement experiments (Cosine Annealing | Warm LR Restartss)
        cos_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0, last_epoch=-1, verbose=False)
        return [opt], []


    def initialize_Gi_LightEffects(self, img, batch_idx):
        # make sure to pass in normalized numpy image (H, W, C) format
        
        # check if img is normalized
        if np.max(img) > 1:
            img = img / 255.0

        # TODO: Retrieve corresponding image from saved Gi, Jinit dirs
         
        return Gi, J_init # note that gi can be viewed properly after multiplying with 255


    def initialize_Li_Luminance(self, img):
        # return the maximum of the three chaneels of img
        return torch.max(img, dim=1)
    

    def initialization_loss(self, G, Gi, L = None, Li=None):
        loss = F.l1_loss(G, Gi)
        if L is not None:
            loss = loss + F.l1_loss(L, Li)
        return loss


    def gradient(self, pred):
        # bs x channels x h x w
        D_dy      = pred[:, :, 1:] - pred[:, :, :-1] # Difference along height
        D_dx      = pred[:, :, :, 1:] - pred[:, :, :, :-1] # Difference along width
        return D_dx, D_dy


    def smooth_loss(self, pred_map):
        dx, dy      = self.gradient(pred_map)
        dx2, dxdy   = self.gradient(dx)
        dydx, dy2   = self.gradient(dy)
        loss        =  (dx2.abs().mean()  + dxdy.abs().mean() + 
                        dydx.abs().mean() + dy2.abs().mean())
        return loss


    def get_exclusion_loss(self, pred, target):
        return self.lambda_excl * self.excl_loss(pred, target)

    
    def get_color_constancy_loss(self, J_init):
        # Gray world assumption
        J_cc = torch.mean(J_init, dim=1, keepdims=True)           
        # Color Constancy Loss 
        cc_loss     = (F.l1_loss(J_init[:, 0:1, :, :], J_cc) + \
                         F.l1_loss(J_init[:, 1:2, :, :], J_cc) + \
                         F.l1_loss(J_init[:, 2:3, :, :], J_cc))*(1/3)
        return self.lambda_cc * cc_loss
    

    def get_reconstruction_loss(self, J_init, G, img):
        recon_loss = F.l1_loss(J_init + G, img)
        return self.lambda_recon * recon_loss


    def get_smooth_loss(self, G):
        smooth_loss = self.smooth_loss(G)
        return self.lambda_smooth * smooth_loss
    
    
    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        iters = 0
        batch_loss = 0.
        for img in tqdm(batch): 
            G = self.forward(img)
            # NOTE: that the authors do NOT provide code for the Shading and Reflectance Networks
            L = None

            # Step 1: Initialize Li and Gi
            Li = self.initialize_Li_Luminance(img)
            Gi, J_init = self.initialize_Gi_LightEffects(img, batch_idx)

            # Step 2: Get gradient exclusion loss
            loss_excl = self.get_exclusion_loss(Gi, J_init)

            # Step 3: Get color constancy loss
            loss_cc = self.get_color_constancy_loss(J_init)

            # Step 4: Get reconstruction loss
            loss_recon = self.get_reconstruction_loss(J_init, G, img)

            # Step 5: Get smoothness loss
            loss_smooth = self.get_smooth_loss(G)

            total_loss = loss_excl + loss_cc + loss_recon + loss_smooth
            batch_loss += total_loss
            
            if iters % len(batch) == 0:
                self.logger.log_image(key="LED-Net_Training", 
                                  images=[Gi, G, J_init, img],
                                  caption=["Gi", "G", "J_init", "I"])
            iters += 1

        self.log('train_loss', batch_loss / len(batch))

        return batch_loss / len(batch)
    

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        results_le = self(batch)
        self.logger.log_image(key="LED-Net_Prediction", images=[make_grid(batch), make_grid(results_le)], 
                              caption=["Input Batch", "Predicted Light Effects Grid"])
        return results_le


def get_LFHF(image, rad_list=[4, 8, 16, 32], eps_list=[0.001, 0.0001]):
    """
        Return Low and High Frequency features' img for HF-loss: Preserves edges & other high
        frequency details of the image (AdaILN might wash stuff away) 
    """
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
        self.gray_scale = keras_cv.layers.Grayscale()


    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(F.conv2d(x, self.image), F.conv2d(x, self.blur))


class LIT_LESLER(pl.LightningModule):
    '''
        TODO: Implement Adaptive Gray from RGB
    '''

    def __init__(self, generatorAB, generatorBA, dGA, dLA, dGB, dLB):
        super().__init__()
        self.save_hyperparameters(ignore=['generatorAB', 'generatorBA', 'dA', 'dB'])

        # No Auto-optimization for GANs 
        self.automatic_optimization = False

        self.genA2B = generatorAB
        self.genB2A = generatorBA
    
        self.disGA = dGA
        self.disLA = dLA
        self.disGB = dGB
        self.disLB = dLB

        # Loss initializations
        self.perceptual_loss = PerceptualLossVgg16ExDark(load_model='./VGG_code/ckpts/vgg16_featureextractFalse_ExDark/nets/model_best.pt')
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.std_loss = StdLoss() # TODO: What is this loss for? (It computes how far the real image is
        # from it's 5x5 blurred version. It's a measure of sharpness.)


    def configure_optimizers(self):
        optG = optim.Adam(chain(self.generator_AB.parameters(), self.generator_BA.parameters()), 
                          lr=0.0001, betas=(0.9, 0.999))
        optD = optim.Adam(chain(self.discriminator_A.parameters(), self.discriminator_B.parameters()),
                                       lr=0.0001, betas=(0.9, 0.999))
        
        # Cosine annealing for Generator and discriminator. Add more params.
        schG = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=10, eta_min=0.00001, last_epoch=-1, verbose=False)
        schD = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=10, eta_min=0.00001, last_epoch=-1, verbose=False)

        # TODO: Experiment with scheduler as well!
        return [optG, optD], []       
     

    def forward(self,x):
        return self.generator_AB(x)

    
    # TODO: Implement: Adaptively fused grayscale conversion of RGB image
    def adaGray(self, img):
        return img
    

    # TODO: Implementation of HF loss (structure consistency)
    def get_HF_loss(self, real, fake):
        pass


    def training_step(self, batch, batch_idx):
        optG, optD = self.optimizers()

        # NOTE: These should be real_X = concat(Jinit_x, G_x)
        # TODO: Will need to make the dataloader as per note above 
        real_A, real_B = batch['real_A'], batch['real_B']

        # TODO: Get adaptively fused Grayscale Images
        real_A_gray, real_B_gray = self.get_adaGray(real_A), self.get_adaGray(real_B)

        # Update D
        optD.zero_grad()

        fake_A2B, _, _ = self.genA2B(real_A)
        fake_B2A, _, _ = self.genB2A(real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)


        # Adversarial Attention Loss (CAM)
        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
        D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
        D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
        
        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
        D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
        D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

        D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
        
        Discriminator_loss = D_loss_A + D_loss_B
        
        self.manual_backward(Discriminator_loss)
        optD.step()


        # Update G
        optG.zero_grad()

        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)
        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)
        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        # Adversarial Attention Loss (CAM)
        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))
        
        # Reconstruction loss
        G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
        G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)
        
        # Identity loss
        G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
        G_identity_loss_B = self.L1_loss(fake_B2B, real_B)
        
        # Purely Attention based loss (For G -> Intuition: CAM will try to classsify into real & fake domains)
        # NOTE: Hence, domain level classifier (add to paper's methodology explanation)
        G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
        G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))
        
        # L_gray_feat: Gray Losses: VGG16 + HF structure consistency loss
        gray_vgg_loss_A = self.perceptual_loss(fake_A2B, real_A_gray)
        gray_hf_loss_A = self.get_HF_loss(fake_A2B, real_A_gray)

        gray_vgg_loss_B = self.perceptual_loss(fake_B2A, real_B_gray)
        gray_hf_loss_B = self.get_HF_loss(fake_B2A, real_B_gray)

        G_loss_A =  gray_vgg_loss_A + gray_hf_loss_A + \
            self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + \
                    self.cam_weight * G_cam_loss_A
        
        G_loss_B = gray_vgg_loss_B + gray_hf_loss_B + \
            self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + \
                    self.cam_weight * G_cam_loss_B
        
        Generator_loss = G_loss_A + G_loss_B

        self.manual_backward(Generator_loss)
        optG.step()

        # clip parameter of AdaILN and ILN, applied after optimizer step
        self.genA2B.apply(self.Rho_clipper)
        self.genB2A.apply(self.Rho_clipper)

        # TODO: Log Losses & Images to WandB here

        

def get_arguments():
    desc = "Pytorch implementation of NightImageEnhancement-2"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_dir', type=str, default='./datasets/', help='Path to the data directory.')
    parser.add_argument('--save_G_dir', type=str, default='./datasets/', help='Path to the directory to save the light-effects map.')
    parser.add_argument('--save_J_dir', type=str, default='./datasets/', help='Path to the directory to save the background image.')

    parser.add_argument('--LED_batch_size', type=int, default=1, help='Batch size for LED-Net.')
    parser.add_argument('--LESLER_batch_size', type=int, default=1, help='Batch size for LESLER-Net.')

    parser.add_argument('--LED_lr', type=float, default=1e-4, help='Learning rate for LED-Net.')
    parser.add_argument('--LESLER_lr', type=float, default=1e-4, help='Learning rate for LESLER-Net.')

    parser.add_argument('--LED_epochs', type=int, default=100, help='Number of epochs for LED-Net.')
    parser.add_argument('--LESLER_epochs', type=int, default=100, help='Number of epochs for LESLER-Net.')

    parser.add_argument('--LED_ckpt_dir', type=str, default='./checkpoints/LED/', help='Path to the directory to save the checkpoints.')
    parser.add_argument('--LESLER_ckpt_dir', type=str, default='./checkpoints/LESLER/', help='Path to the directory to save the checkpoints.')

    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the dataloader.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use. 0: 3090 GeForce GTX | 1: 1080 Ti')

    # for LESLER-Net
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--dataset', type=str, default='Kaggle', help='dataset_name')
    parser.add_argument('--datasetpath', type=str, default='./datasets/kaggle_ds_night_road/night_road/', help='dataset_path')
    parser.add_argument('--iteration', type=int, default=900000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--atten_weight', type=int, default=0.5, help='Weight for Attention Loss')
    parser.add_argument('--use_gray_feat_loss', type=str2bool, default=True, help='use Structure and HF-Features Consistency Losses')
    parser.add_argument('--feat_weight', type=int, default=1, help='Weight for Structure and HF-Features Consistency Losses')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN Loss')
    parser.add_argument('--identity_weight', type=int, default=5, help='Weight for Identity Loss')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=512, help='The training size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results/', help='Directory name to save the training results')
    parser.add_argument('--benchmark_flag', type=str2bool, default=True)
    parser.add_argument('--resume', type=str2bool, default=True)

    parser.add_argument('--model_name', type=str, default='LOL_params_0900000.pt', help='model name to load')
    parser.add_argument('--out_dir', type=str, default='', help='Directory name to save the inference results')
    
    parser.add_argument('--im_suffix', type=str, default='.png', help='The suffix of test images [.png / .jpg]')

    args = parser.parse_args()

    return args
    

def load_LED_initialized_data(base_dir, save_G_dir, save_J_dir):
    LED_data = LED_init_Dataset(base_dir, save_G_dir, save_J_dir)
    LED_dataloader = DataLoader(LED_data, 
                                batch_size=1, 
                                shuffle=True, 
                                num_workers=8)
    return LED_dataloader # Don't really need this dataloader tbh. Just need the saved outputs' location.


def load_LESLER_data(args):
    if args.phase == 'train':
        transform = T.Compose([T.Resize(args.img_size, 
                                        interpolation=T.InterpolationMode.BILINEAR, 
                                        antialias=True), 
                               T.ToTensor()])
    elif args.phase == 'test':
        transform = T.Compose([T.ToTensor()])
    else:
        print(f"[!] Invalid phase: {args.phase}\nPlease choose between train and test")
        exit(1)

    dataFolder = ImageFolder(args.datasetpath, transform)
    dataLoader = DataLoader(dataFolder, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    return dataLoader


if __name__ == '__main__':
    args = get_arguments()
    print(args)

    LED_dataloader = load_LED_initialized_data(args.data_dir, args.save_G_dir, args.save_J_dir)
    # TODO 1: Call LED-Net
    # TODO 2: Train LED-Net
    # TODO 3: Save the model & it's output

    # TODO4: Load LESLER data
    LESLER_dataloader = load_LESLER_data(args)

    # TODO5 (DONE): Call LESLER-Net networks

    # Generators: Unpaired Image-to-Image Translation (Unsupervised but need set-level supervision)
    genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size)
    genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size)
    
    # Global and local OR inc. complexity of features' Discriminators 
    # (ideally you would want inf Ds with increasing depth) 
    disGA = Discriminator(input_nc=3, ndf=args.ch, n_layers=7) # Global Discriminator (deeper network => Higher level features analyzed)
    disLA = Discriminator(input_nc=3, ndf=args.ch, n_layers=5) # Local Discriminator (shallower network => Lower level features analyzed)
    
    # TODO 5: Train LESLER-Net on LED-Net's output
    # TODO 6: Save the model & it's output
    