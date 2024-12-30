#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:26:37 2024

@author: lillianli

# This code is used to generates a CSV file with two columns: 'file_name' and 'label'.
The script looks for four subfolders under root_dir: Benign, InSitu, Invasive, Normal.
Each subfolder is mapped to a label:
    Benign   -> 1
    InSitu   -> 2
    Invasive -> 3
    Normal   -> 4

It only processes files that end with '.npy', renaming them to '.pny' in the CSV file.

:param root_dir:   Path to the main directory containing the four subfolders.
:param output_csv: Output CSV file name (or path).

The generated csv file can be used for Training set and Testing set split
"""


import os
import csv

def generate_csv(
    root_dir = "/scratch/wang_lab/BRCA_project/Data/BACH_resize_crop_train", # change this to actual directory
    output_csv = "BACH_sample_description_UniquelyMatched.csv" # change this to actual file
):
    folder_labels = {
        "Benign": 1,
        "InSitu": 2,
        "Invasive": 3,
        "Normal": 4
    }
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "label"])
        for folder_name, label in folder_labels.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"Warning: {folder_path} not found or is not a directory.")
                continue

            for file_name in os.listdir(folder_path):
                full_path = os.path.join(folder_path, file_name)
                if os.path.isfile(full_path) and file_name.endswith(".npy"):
                    base_name = os.path.splitext(file_name)[0] + ".pny"
                    writer.writerow([base_name, label])


if __name__ == "__main__":
    generate_csv()