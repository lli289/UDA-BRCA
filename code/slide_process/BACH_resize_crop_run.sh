#!/bin/bash                                                                     
                                                    
#SBATCH --partition=stats_medium                   #Name of the partition  
#SBATCH --job-name BACH_resize_crop                     #Nameof the job         
#SBATCH --output output/job_output_%j.txt                #Output file name
#SBATCH --error output/job_error_%j.txt        
#SBATCH --ntasks=1                                     #Number of cores        
#SBATCH --mem=16G
#SBATCH --mail-type ALL                                                         
#SBATCH --mail-user jxu238@uky.edu                      #Email to forward       
#SBATCH --time=3-00:00:00                                

source /home/jxu238/miniconda3/etc/profile.d/conda.sh
conda activate brca

python BACH_crop_rescale_testset.py --input_dir=/scratch/wang_lab/BRCA_project/Data/BACH/ICIAR2018_BACH_Challenge_TestDataset/Photos --output_dir=/scratch/wang_lab/BRCA_project/Data/BACH/ICIAR2018_BACH_Challenge_TestDataset/Photos_cr