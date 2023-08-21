#!/usr/bin/env python

import cv2
import os
import numpy as np
import glob
from tqdm.auto import tqdm

img_array = []

REAL = "./datasets/kaggle_ds_night_road/night_road/"
FAKE_deli = "./output/kaggle_deli/"
FAKE_LOL = "./output/kaggle_LOL/"

def export_vid(FAKE_deli, name):
    for filename in tqdm(sorted(os.listdir(FAKE_deli))):
        img = cv2.imread(FAKE_deli+filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(f'{name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
 
    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])

    out.release()

# export_vid(FAKE_deli, "fake_deli")
export_vid(FAKE_LOL, "fake_LOL")
# export_vid(REAL, "REAL")