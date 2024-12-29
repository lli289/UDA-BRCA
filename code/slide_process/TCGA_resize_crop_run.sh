#!/bin/bash                                                                     
                                                    
#SBATCH --partition=stats_medium                    #Name of the partition  
#SBATCH --job-name tcga_resize_crop                     #Nameof the job         
#SBATCH --output output/job_output_%j.txt                #Output file name
#SBATCH --error output/job_error_%j.txt        
#SBATCH --ntasks=1                                     #Number of cores        
#SBATCH --mem=16G
#SBATCH --mail-type ALL                                                         
#SBATCH --mail-user jxu238@uky.edu                      #Email to forward       
#SBATCH --time=3-00:00:00                                

source /home/jxu238/miniconda3/etc/profile.d/conda.sh
conda activate brca

python TCGA_rescale_crop.py --input_dir=/scratch/wang_lab/BRCA_project/Data/TCGA_original/TCGA_BRCA --coords_file=/scratch/wang_lab/BRCA_project/Data/TCGA_original/TCGA_BRCA_tile_coordinates.pkl --output_dir=/scratch/wang_lab/BRCA_project/Data/TCGA_resize_cr