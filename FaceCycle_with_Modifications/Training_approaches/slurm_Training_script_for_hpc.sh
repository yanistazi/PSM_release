#! /bin/bash
#SBATCH --job-name=Training_approach
## Number of nodes
#SBATCH --nodes=4 # 4 GPU Nodes
## Number of of cpus required for the task
#SBATCH --cpus-per-task=16
## Number of tasks
##SBATCH --ntasks=16
#SBATCH -o out_slurm_folder/train_expression_%x%j.out    # STDOUT
#SBATCH -e error_slurm_folder/train_expression_%x%j.err    # STDERR
#SBATCH --partition=hpc_v100  # GPU Name
##SBATCH --exclusive
echo Starting at `date`
echo This is job $SLURM_JOB_ID
echo Running on `hostname`
source miniconda3/etc/profile.d/conda.sh
conda activate PSM_env
CUDA_VISIBLE_DEVICES=0,1 python python Training_Approaches_Expression.py 
echo Exiting at `date`