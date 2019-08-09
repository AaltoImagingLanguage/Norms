#!/bin/bash

# For testing, comment for main analysis
# SLURM_ARRAY_TASK_ID=5

# Make sure to request only the resources you really need to avoid cueing
#SBATCH --time=0:05:00
#SBATCH --mem-per-cpu=2G
#SBATCH -n 1

# Do 1000 random iterations. 
#SBATCH --array=1-1000

# The input files for all the subjects. Make sure the path is correct!
INPUT_FILES=( 
    cslb
    w2v_eng
)

cd /m/nbe/work/kivisas1/aaltonorms
# The directory in which to place the results. Make sure the path is correct!
ROOT_PATH=/m/nbe/scratch/aaltonorms/results/zero_shot/perm/
OUTPUT_PATH="$ROOT_PATH${INPUT_FILES[0]}_${INPUT_FILES[1]}"

# Make sure the output path exists
mkdir -p $OUTPUT_PATH

# Construct the names of the output files of this run
LOG_FILE=$OUTPUT_PATH/$(printf 's%02d_results.out' $SLURM_ARRAY_TASK_ID)
OUTPUT_FILE=$OUTPUT_PATH/$(printf 'iteration_%04d_results.mat' $SLURM_ARRAY_TASK_ID)

# On triton, uncomment this to load the Python environment. On taito, you
# presumably installed Python yourself.
module load anaconda3



# Run the analysis
srun -o $LOG_FILE python zero_shot_decoding_leave1out_perm.py ${INPUT_FILES[*]} -i $SLURM_ARRAY_TASK_ID -o $OUTPUT_FILE
