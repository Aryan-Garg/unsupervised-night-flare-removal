#!/usr/bin/env python

import os
import sys

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
import matplotlib.pyplot as plt

# Custom imports
from decomposition_code import decompose

class LED_init_Dataset(Dataset):
    def __init__(self, base_dir, save_G_dir, save_J_dir, transform=Compose([ToTensor()])):
        super().__init__()
        self.base_dir = base_dir
        self.transform = transform
        self.save_G_dir = save_G_dir
        self.save_J_dir = save_J_dir


    def __len__(self):
        return len(os.listdir(self.base_dir))


    def __getitem__(self, index):
        # Call the matlab scripts to get the images decomposed and saved 
        for filename in os.listdir(self.base_dir + ""):
            decompose.decompose_image(filename, self.base_dir+"", self.save_G_dir+"", self.save_J_dir+"")
        
        # Now read the indexed image from G, base and Jinit folders and return them in a dict
        # NOTE: All dirs have same filenames! :p
        
        # print(index)
        # Read the image from G folder
        file_G = os.listdir(self.save_G_dir)[index]

        img_G = cv2.imread(self.save_G_dir + file_G)
        img_J = cv2.imread(self.save_J_dir + file_G)
        img = cv2.imread(self.base_dir + file_G)

        return {'I': img, 'Gi': img_G, 'Jinit': img_J}    

