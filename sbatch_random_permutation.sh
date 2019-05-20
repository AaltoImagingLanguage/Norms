#!/bin/bash

# SLURM_ARRAY_TASK_ID=5

# Make sure to request only the resources you really need to avoid cueing
#SBATCH --time=0:05:00
#SBATCH --mem-per-cpu=2G
#SBATCH -n 1

# Do 1000 random iterations. 
#SBATCH --array=1-1000

# The input files for all the subjects. Make sure the path is correct!
INPUT_FILES=( 
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG001_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG002_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG003_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG004_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG005_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG006_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG007_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG008_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG009_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG010_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG011_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG012_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG014_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG015_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG016_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG017_top500_cue3_LUT.mat
    /m/nbe/scratch/guessfmri/machineLearning/data/topVoxels/GG018_top500_cue3_LUT.mat

)


# Location of the semantic feature norms. Make sure the path is correct!
DATAPATH='/m/nbe/scratch/guessfmri/machineLearning/data/'
NORM_FILE="${DATAPATH}corpusvectors_ginter_lemma.mat"

# The directory in which to place the results. Make sure the path is correct!
OUTPUT_PATH='/m/nbe/scratch/guessfmri/machineLearning/results/perm/stabsel_target'

# Make sure the output path exists
mkdir -p $OUTPUT_PATH

# Construct the names of the output files of this run
LOG_FILE=$OUTPUT_PATH/$(printf 's%02d_results.out' $SLURM_ARRAY_TASK_ID)
OUTPUT_FILE=$OUTPUT_PATH/$(printf 'iteration_%04d_results.mat' $SLURM_ARRAY_TASK_ID)

# On triton, uncomment this to load the Python environment. On taito, you
# presumably installed Python yourself.
module load anaconda3
source activate /m/nbe/scratch/guessfmri/machineLearning/envs/my_root

# Run the analysis
srun -o $LOG_FILE python run_zero_shot_permutation.py -v --norms $NORM_FILE -i $SLURM_ARRAY_TASK_ID -o $OUTPUT_FILE ${INPUT_FILES[*]} -d "cosine"
