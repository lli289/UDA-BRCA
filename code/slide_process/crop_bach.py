import os
import norm as nm
import numpy as np
import glob
import argparse

from PIL import Image


def crop_tiles(image, tile_size, stride):
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, stride):
        if y + tile_size > h:
            y = h - tile_size
        for x in range(0, w, stride):
            if x + tile_size > w:
                x = w - tile_size
            tiles.append(image[y:y + tile_size, x:x + tile_size])
    return tiles



def process_and_save_images(input_folder, output_folder, tile_size, stride):
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_folder, class_folder)
            os.makedirs(output_class_path, exist_ok=True)

            for img_file in glob.glob(os.path.join(class_path, "*.tif")):
                img_name = os.path.basename(img_file).replace('.tif', '')
                img = np.array(Image.open(img_file))
                tiles = crop_tiles(img, tile_size, stride)

                for i, tile in enumerate(tiles, start=1):
                    tile_num = f"{i:02d}"
                    np.save(os.path.join(output_class_path, f"{img_name}_{tile_num}_norm.npy"), tile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save images as tiles")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")
    parser.add_argument('--tile_size', type=int, default=224, help="Size of the tiles")
    parser.add_argument('--overlap', type=int, default=32, help="Overlap size for the tiles")

    args = parser.parse_args()

    stride = args.tile_size - args.overlap
    process_and_save_images(args.input_dir, args.output_dir, args.tile_size, stride)