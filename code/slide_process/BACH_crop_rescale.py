#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 22:34:06 2024

@author: lillianli
"""

import os
import numpy as np
import openslide
import argparse
import cv2
import openslide.deepzoom

# Lillian 12/26
# This code is used to rescale and crop BACH histology images

# Notes:
# Original BACH histology images 2048 x 1536 pixel
# Origianl BACH pixel scale 0.42um x 0.42um

# Goal: 
# 1. BACH: Resize original histology images to 512 x 384 resolution 
#       new pixel scale after resizing: 1.68um x 1.68um
# 2. TCGA: Resize original histology images to pixel scale 1.68um x 1.68um
# 3. Crop resized BACH and TCGA to 224 x 224 patches with 50% overlap
    
def rescale_image(image):
    height, width = image.shape[:2]
    new_height, new_width = int(height), int(width) 
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

def save_numpy_tiles(path2slides, folder, slidename, output_path, height, width, strideH, strideW):
    slide_path = os.path.join(path2slides, folder, slidename)
    slide_name, _ = os.path.splitext(os.path.basename(slidename))
    slide = openslide.OpenSlide(slide_path)
    slide_rescaled = rescale_image(slide)  
    slide_crop = crop_image(slide_rescaled, height, width, strideH, strideW)
    for idx, tile in enumerate(slide_crop):
        tile_filename = f"{slide_name}_{idx + 1:04d}.npy"
        tile_path = os.path.join(output_path, tile_filename)
        np.save(tile_path, tile)
        print(f"Saved rescaled tile: {tile_filename}")  

# Process
def process_all_slides(path2slides, output_path, height, width, strideH, strideW):
    slide_dirs = [d for d in os.listdir(path2slides) if os.path.isdir(os.path.join(path2slides, d))]

    slidenames = []
    subfolders = []

    for d in slide_dirs:
        for f in os.listdir(os.path.join(path2slides, d)):
            if (f.endswith('.svs') or f.endswith('.tif')) and 'mask' not in f:
                slidenames.append(f)
                subfolders.append(d)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for folder, slidename in zip(subfolders, slidenames):
        output_folder = os.path.join(output_path, folder)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        save_numpy_tiles(path2slides, folder, slidename, output_folder, height, width, strideH, strideW)

# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BACH slides")
    parser.add_argument('--input_dir', type = str, required = True, help = "Path to the input directory containing BACH files")
    parser.add_argument('--output_dir', type = str, required = True, help = "Path to the output directory to save tiles")
    args = parser.parse_args()
    height = 384
    width = 512
    strideH = 192
    strideW = 256
    process_all_slides(args.input_dir, args.output_dir, height, width, strideH, strideW)