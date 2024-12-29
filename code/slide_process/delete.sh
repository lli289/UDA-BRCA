#!/bin/bash                                                                     
                                                    
#SBATCH --partition=stats_medium                    #Name of the partition  
#SBATCH --job-name delete                    #Nameof the job         
#SBATCH --output output/job_output_%j.txt                #Output file name
#SBATCH --error output/job_error_%j.txt        
#SBATCH --ntasks=1                                     #Number of cores        
#SBATCH --mem=16G
#SBATCH --mail-type ALL                                                         
#SBATCH --mail-user jxu238@uky.edu                      #Email to forward       
#SBATCH --time=3-00:00:00                                


rm -rf /scratch/wang_lab/BRCA_project/DANN/Archive/Data/DANN_ALL/BACH_TCGA/WSI_resize_crop/