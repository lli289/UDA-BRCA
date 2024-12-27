#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:58:23 2024

@author: lillianli
"""
# This code is used to resize and crop TCGA histology images
# Resize to 1.68um/pixel
# Crop using coordinates, no overlap
# Note that these are raw/absolute coordinates corresponding to 0.5mpp tiles
# We don't want 0.5mpp tiles, we want 1.68mpp tiles
# The key is to recale the raw coordinates as well
# Note that for these coordinates, some are level 16, some 15, some 17 they have different levels
# All to level 18, which is very close to 1.68 to 2.0mpp
# We made sure index x and index y all rescaled properly
# Cropped tile size 224 x 224 overlap = 0

import os
import numpy as np
import openslide
import argparse
import pickle
import openslide.deepzoom

from PIL import Image

# Functions from cca package
def stretch_pre(nimg):
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.maximum(nimg[0] - nimg[0].min(), 0)
    nimg[1] = np.maximum(nimg[1] - nimg[1].min(), 0)
    nimg[2] = np.maximum(nimg[2] - nimg[2].min(), 0)
    return nimg.transpose(1, 2, 0)

def max_white(nimg):
    if nimg.dtype == np.uint8:
        brightest = float(2 ** 8)
    elif nimg.dtype == np.uint16:
        brightest = float(2 ** 16)
    elif nimg.dtype == np.uint32:
        brightest = float(2 ** 32)
    else:
        brightest = float(2 ** 8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest / float(nimg[0].max())), 255)
    nimg[1] = np.minimum(nimg[1] * (brightest / float(nimg[1].max())), 255)
    nimg[2] = np.minimum(nimg[2] * (brightest / float(nimg[2].max())), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def stretch(nimg):
    return max_white(stretch_pre(nimg))

def from_pil(pimg):
    pimg = pimg.convert(mode='RGB')
    nimg = np.array(pimg)
    return nimg

def to_pil(nimg):
    return Image.fromarray(np.uint8(nimg))

def transform_coords_for_level18(coords):
    new_coords = []
    for row in coords:
        old_level = int(row[0])
        x = int(row[1])
        y = int(row[2])
        delta = 18 - old_level

        if delta > 0:
            scale_factor = 2 ** delta
            new_x = x // scale_factor
            new_y = y // scale_factor
        elif delta < 0:
            scale_factor = 2 ** abs(delta)
            new_x = x * scale_factor
            new_y = y * scale_factor
        else:
            new_x = x
            new_y = y
        new_coords.append([18, new_x, new_y])
    return np.array(new_coords, dtype=np.int64)

def extract_tile_features(coord, zoom):
    level = coord[0]  
    tile_x = coord[1]
    tile_y = coord[2]
    tile = zoom.get_tile(level, (tile_x, tile_y))
    tile = np.array(tile)
    tile = to_pil(stretch(from_pil(Image.fromarray(tile))))
    return np.array(tile)

def save_numpy_tiles(path2slides, folder, slidename, coords, output_path):
    slide_path = os.path.join(path2slides, folder, slidename)
    slide_name = os.path.splitext(os.path.basename(slidename))[0]
    slide = openslide.OpenSlide(slide_path)
    zoom = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=0)
    new_coords = transform_coords_for_level18(coords)
    level = 18
    max_tiles_x, max_tiles_y = zoom.level_tiles[level]  # e.g. (num_tiles_x, num_tiles_y)
    for idx, new_coord in enumerate(new_coords):
        _, tile_x, tile_y = new_coord
        if 0 <= tile_x < max_tiles_x and 0 <= tile_y < max_tiles_y:
            tile = extract_tile_features(new_coord, zoom)
            tile_filename = f"{slide_name}_{idx+1:04d}.npy"
            tile_path = os.path.join(output_path, tile_filename)
            np.save(tile_path, tile)
            print(f"Saved tile {tile_filename} at level=18 (≈1.68 mpp)")
        else:
            print(f"Warning: tile index out of range at {new_coord}, skipping...")

def process_all_slides(path2slides, tile_coords, output_path):
    slide_dirs = [d for d in os.listdir(path2slides)
                  if os.path.isdir(os.path.join(path2slides, d))]

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
        if slidename in tile_coords.keys():
            output_folder = os.path.join(output_path, folder)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            save_numpy_tiles(path2slides, folder, slidename, tile_coords[slidename],
                             output_folder)
        else:
            print(f'Warning: tile coordinates not found for file {slidename}, skipping.')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WSI tiles at ~1.68 mpp.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Path to input directory containing subfolders with WSI slides.")
    parser.add_argument('--coords_file', type=str, required=True,
                        help="Pickle file with tile coords (shape Nx3: [level, x, y]).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory for extracted tiles.")
    args = parser.parse_args()
    with open(args.coords_file, 'rb') as f:
        tile_coords = pickle.load(f)
    process_all_slides(args.input_dir, tile_coords, args.output_dir)
