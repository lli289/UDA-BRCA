import os
import numpy as np
import openslide
import argparse
import pickle
import cv2
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

# Function to extract and process tiles
def extract_tile_features(level, coord, zoom):
    tile = np.array(zoom.get_tile(level, (coord[1], coord[2])))
    tile = Image.fromarray(tile)
    tile = to_pil(stretch(from_pil(tile)))
    tile = np.array(tile)
    return tile

def rescale_image(image, scale_factor):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def save_numpy_tiles(path2slides, folder, slidename, coords, output_path, scale_factor):
    slide_path = os.path.join(path2slides, folder, slidename)
    slide_name = os.path.basename(slidename).replace('.svs', '')
    slide = openslide.OpenSlide(slide_path)
    zoom = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=0)
    level = int(coords[0, 0])

    for idx, coord in enumerate(coords):
        tile = extract_tile_features(level, coord, zoom)
        tile_rescaled = rescale_image(tile, scale_factor)
        tile_filename = f"{slide_name}_{idx + 1:04d}.npy"
        tile_path = os.path.join(output_path, tile_filename)
        np.save(tile_path, tile_rescaled)
        print(f"Saved rescaled tile {tile_filename}")

def process_all_slides(path2slides, tile_coords, output_path, scale_factor):
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
        if slidename in tile_coords.keys():
            output_folder = os.path.join(output_path, folder)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            save_numpy_tiles(path2slides, folder, slidename, tile_coords[slidename], output_folder, scale_factor)
        else:
            print(f'Warning: tile coordinates not found for file {slidename}, skipping it')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TCGA WSI slides")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory containing WSI files")
    parser.add_argument('--coords_file', type=str, required=True, help="Path to the file containing tile coordinates")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory to save tiles")

    args = parser.parse_args()

    # Load tile coordinates
    with open(args.coords_file, 'rb') as f:
        tile_coords = pickle.load(f)

    scale_factor = 0.42 / 0.5
    process_all_slides(args.input_dir, tile_coords, args.output_dir, scale_factor)