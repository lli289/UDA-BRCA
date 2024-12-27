import os
import numpy as np
import openslide
import argparse
import cv2
import openslide.deepzoom
import pandas as pd

from PIL import Image

# Lillian 12/26
# This code is used to rescale and crop TCGA histology images

# Rescale part:
# TCGA Histology images 2048 x 1536 pixel
# BACH Histology images 2048 x 1536 pixel
# BACH Pixel scale 0.42 µm x 0.42 µm
# scale_factor = 0.42 / data['Scale.X'] e.g.,
# each one divided by it's own scale, not just one fixed number!

# Crop part:
# From left to right top and bottom

# Define functions part:
    
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
    
def rescale_image(image, scale_factor):
    height, width = image.shape[:2]
    new_height, new_width = int(height / scale_factor), int(width / scale_factor) # Lillian edited on 12/16 divide not multiply
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

def save_numpy_tiles(path2slides, folder, slidename, output_path, scale_factor, height, width, strideH, strideW):
    slide_path = os.path.join(path2slides, folder, slidename)
    slide_name = os.path.basename(slidename).replace('.svs', '')
    slide = openslide.OpenSlide(slide_path)
    slide_rescaled = rescale_image(slide, scale_factor)
    slide_crop = crop_image(slide_rescaled, height, width, strideH, strideW)
    
    for idx, tile in enumerate(slide_crop):
        tile_filename = f"{slide_name}_{idx + 1:04d}.npy"
        tile_path = os.path.join(output_path, tile_filename)
        np.save(tile_path, tile)
        print(f"Saved rescaled tile {tile_filename}")
        
# Process
def process_all_slides(path2slides, output_path, height, width, strideH, strideW, scale_data):
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
        # Each one divided by it's on scale; not a fixed number!
        # Remember to match!
        scale_factor = 0.42 / scale_data['Scale.X'][scale_data['Slide.ID'] == slidename]
        save_numpy_tiles(path2slides, folder, slidename, output_folder, scale_factor, height, width, strideH, strideW)
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TCGA WSI slides")
    parser.add_argument('--input_dir', type = str, required = True, help = "Path to the input directory containing WSI files")
    parser.add_argument('--output_dir', type = str, required = True, help = "Path to the output directory to save tiles")
    args = parser.parse_args()
    data = pd.read_csv('TCGA_scale.csv') # !!!!!!!change this working directory to the actual file location!
    height = 1536
    width = 2048
    strideH = 384
    strideW = 512
    process_all_slides(args.input_dir, args.output_dir, height, width, strideH, strideW, data)