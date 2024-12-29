#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:46:58 2024

@author: lillianli
"""

import os
import numpy as np
import argparse
import cv2
import glob

from PIL import Image

# Lillian 12/26
# This code is used to rescale and crop BACH histology images 
# Use this code on the BACH histology images Testing set

# Notes:
# Original BACH histology images 2048 x 1536 pixel
# Original BACH pixel scale 0.42um x 0.42um

# Goal: 
# 1. BACH: Resize original histology images to 512 x 384 resolution 
#       new pixel scale after resizing: 1.68um x 1.68um
# 3. Crop resized BACH to 224 x 224 patches with 50% overlap
    
def rescale_image(image, target_scale=1.66, original_scale=0.42):
    scale_factor = original_scale / target_scale
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def crop_image(image, height, width, strideH, strideW):
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, strideH):
        if y + height > h:
            y = h - height
        for x in range(0, w, strideW):
            if x + width > w:
                x = w - width
            tiles.append(image[y:y + height, x:x + width])
    return tiles

def save_numpy_tiles(input_folder, output_folder, height, width, strideH, strideW):
    os.makedirs(output_folder, exist_ok = True)
    
    for img_file in glob.glob(os.path.join(input_folder, "*.tif")):
        img_name = os.path.basename(img_file).replace('.tif', '')  
        img = np.array(Image.open(img_file))  
        img_rescaled = rescale_image(img)  
        tiles = crop_image(img_rescaled, height, width, strideH, strideW)  

        for idx, tile in enumerate(tiles, start=1):
            tile_filename = f"{img_name}_{idx:03d}.npy"
            tile_path = os.path.join(output_folder, tile_filename)
            np.save(tile_path, tile)

# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BACH slides")
    parser.add_argument('--input_dir', type = str, required = True, help = "Path to the input directory containing BACH files")
    parser.add_argument('--output_dir', type = str, required = True, help = "Path to the output directory to save tiles")
    args = parser.parse_args()
    height = 224
    width = 224
    strideH = 112  # 50% overlap
    strideW = 112  # 50% overlap
    save_numpy_tiles(args.input_dir, args.output_dir, height, width, strideH, strideW)
