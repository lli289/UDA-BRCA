import os

# Define the root paths for HPC
hpc_root = "/scratch/wang_lab/BRCA_project/data"
target_root = "TCGA/WSI_resize_crop"

# Define the label for all files
label = 0

# Define the output file
output_file = "file_paths_labels.txt"

# Open the output file
with open(output_file, "w") as f:
    # Walk through each folder and file in the target directory
    for root, dirs, files in os.walk(os.path.join(hpc_root, target_root)):
        for file in files:
            # Get the full path
            full_path = os.path.join(root, file)

            # Create the relative path for the file
            relative_path = os.path.relpath(full_path, hpc_root).replace("\\", "/")

            # Write the relative path and label to the output file
            f.write(f"{relative_path} {label}\n")

print(f"File paths and labels written to {output_file}")
