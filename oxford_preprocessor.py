#!/usr/bin/env python

import os
from tqdm.auto import tqdm
import numpy as np
import cv2

DATA_DIR =  './datasets/2014-11-14-16-34-33/stereo/right/'
SAVE_DIR = './datasets/oxford/'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
else:
    # remove all files in the directory
    for file in os.listdir(SAVE_DIR):
        os.remove(SAVE_DIR + file)

# take in raw bayer images and convert to rgb
def bayer_to_rgb(bayer):
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerGBRG2BGR)
    return rgb


for file in tqdm(os.listdir(DATA_DIR)):
     if file.endswith(".png"):
        bayer = cv2.imread(DATA_DIR + file, cv2.IMREAD_UNCHANGED)
        rgb = bayer_to_rgb(bayer)
        cv2.imwrite(SAVE_DIR + file, rgb)

