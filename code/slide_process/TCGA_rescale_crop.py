#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:58:23 2024

@author: lillianli

This code is used to resize and crop TCGA histology images to ~1.68 μm/pixel.
We handle the possibility that slides differ in base resolution, by dynamically
computing the ideal level.

The coordinates used for cropping are raw/absolute corresponding to 0.5mpp tiles
We don't want 0.5mpp, we want 1.68 mpp tiles. We need to rescale coordinates as well.
Final cropped tile size 224 x 224 with no overlap.
"""

import os
import numpy as np
import openslide
import argparse
import pickle
import openslide.deepzoom
import math

from PIL import Image

##############################################################################
# Color/contrast helper functions
##############################################################################
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
    nimg = nimg.transpose(2, 0, 1).astype(np.int32)
    for i in range(3):
        max_val = float(nimg[i].max()) if nimg[i].max() != 0 else 1.0
        nimg[i] = np.minimum(nimg[i] * (brightest / max_val), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def stretch(nimg):
    return max_white(stretch_pre(nimg))

def from_pil(pimg):
    pimg = pimg.convert(mode='RGB')
    return np.array(pimg)

def to_pil(nimg):
    return Image.fromarray(np.uint8(nimg))

##############################################################################
# Transform coordinates from (old_level, x, y) to (new_level, x', y')
##############################################################################
def transform_coords(coords, new_level):
    """
    coords shape: (N, 3), each row = [old_level, x, y]
    new_level: the level to transform to
    """
    new_coords = []
    for row in coords:
        old_level = int(row[0])
        x = int(row[1])
        y = int(row[2])
        delta = new_level - old_level

        if delta > 0:
            # Going deeper in the pyramid => use integer division
            scale_factor = 2 ** delta
            new_x = x // scale_factor
            new_y = y // scale_factor
        elif delta < 0:
            # Going to a finer level => multiply
            scale_factor = 2 ** abs(delta)
            new_x = x * scale_factor
            new_y = y * scale_factor
        else:
            # Same level => no change
            new_x = x
            new_y = y

        new_coords.append([new_level, new_x, new_y])
    return np.array(new_coords, dtype=np.int64)

##############################################################################
# Extract a tile at a given (level, x, y)
##############################################################################
def extract_tile_features(coord, zoom):
    level = coord[0]
    tile_x = coord[1]
    tile_y = coord[2]

    tile = zoom.get_tile(level, (tile_x, tile_y))  # returns a PIL Image
    tile = np.array(tile)
    # Apply color normalization
    tile = to_pil(stretch(from_pil(Image.fromarray(tile))))
    return np.array(tile)

##############################################################################
# Save tiles for one slide
##############################################################################
def save_numpy_tiles(path2slides, folder, slidename, coords, output_path):
    """
    1) Open the slide
    2) Compute the "ideal" level for ~1.68 mpp from base_mpp in the metadata
    3) Clamp that level to valid range [0, zoom.level_count-1]
    4) Transform old coords => new coords at the chosen level
    5) For each tile index, check bounds. If valid, extract & save. Else skip.
    """
    slide_path = os.path.join(path2slides, folder, slidename)
    slide_name = os.path.splitext(os.path.basename(slidename))[0]

    # 1) Open slide
    slide = openslide.OpenSlide(slide_path)
    zoom = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=0)

    # 2) Compute ideal level
    base_mpp = float(slide.properties.get('openslide.mpp-x', 0.25))  # fallback
    desired_mpp = 1.68
    # If base_mpp is 0.25, log2(1.68 / 0.25) => ~2.75 => round => level=3
    ideal_level_float = math.log2(desired_mpp / base_mpp)
    ideal_level = int(round(ideal_level_float))

    # 3) Clamp the ideal level to what the slide actually has
    max_level_index = zoom.level_count - 1  # largest valid pyramid index
    if ideal_level < 0:
        ideal_level = 0
    elif ideal_level > max_level_index:
        ideal_level = max_level_index

    # 4) Transform coords from old_level -> ideal_level
    new_coords = transform_coords(coords, ideal_level)

    # 5) Check tile bounds at that level
    tiles_x, tiles_y = zoom.level_tiles[ideal_level]

    for idx, new_coord in enumerate(new_coords):
        _, tile_x, tile_y = new_coord
        if 0 <= tile_x < tiles_x and 0 <= tile_y < tiles_y:
            tile = extract_tile_features(new_coord, zoom)
            tile_filename = f"{slide_name}_{idx+1:04d}.npy"
            tile_path = os.path.join(output_path, tile_filename)
            np.save(tile_path, tile)
            print(f"Saved tile {tile_filename} at level={ideal_level} (~{desired_mpp} mpp)")
        else:
            print(f"Warning: tile index out of range at {new_coord}, skipping...")

##############################################################################
# Process all slides
##############################################################################
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
            save_numpy_tiles(path2slides, folder, slidename, tile_coords[slidename], output_folder)
        else:
            print(f'Warning: tile coordinates not found for file {slidename}, skipping...')

##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WSI tiles at ~1.68 mpp.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Path to input directory containing subfolders with WSI slides.")
    parser.add_argument('--coords_file', type=str, required=True,
                        help="Pickle file with tile coords (shape Nx3: [level, x, y]).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory for extracted tiles.")

    args = parser.parse_args()

    # Load tile coordinates (dict: {slidename: np.array of shape Nx3})
    with open(args.coords_file, 'rb') as f:
        tile_coords = pickle.load(f)

    process_all_slides(args.input_dir, tile_coords, args.output_dir)
