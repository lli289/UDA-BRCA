import os

# Define the root paths for local and HPC
local_root = "E:/Project_data/BRCA_Project/BACH/ICIAR2018_BACH_Challenge/Photos_crop"
hpc_base_root = "/scratch/wang_lab/BRCA_project/data"
hpc_relative_root = "BACH/ICIAR2018_BACH_Challenge/Photos_crop"

# Define the mapping of folder names to labels
label_map = {
    "Normal": 0,
    "Benign": 1,
    "InSitu": 2,
    "Invasive": 3
}

# Define the output file
output_file = "bach_labels.txt"

# Open the output file
with open(output_file, "w") as f:
    # Walk through each folder and file in the local root directory
    for root, dirs, files in os.walk(local_root):
        for file in files:
            # Get the full local path
            local_path = os.path.join(root, file)

            # Create the relative path for HPC by replacing local_root with hpc_relative_root
            relative_path = local_path.replace(local_root, hpc_relative_root).replace("\\", "/")

            # Determine the label based on the parent folder name
            parent_folder = os.path.basename(os.path.dirname(local_path))
            label = label_map.get(parent_folder, None)

            if label is not None:
                # Write the relative path and label to the output file
                f.write(f"{relative_path} {label}\n")

print(f"File paths and labels written to {output_file}")
