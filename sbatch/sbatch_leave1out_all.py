#!/bin/bash

# For testing, comment for main analysis
# SLURM_ARRAY_TASK_ID=5

# Make sure to request only the resources you really need to avoid cueing
#SBATCH --time=0:05:00
#SBATCH --mem-per-cpu=2G
#SBATCH -n 1


# Construct the names of the output files of this run
OUTPUT_PATH=/m/nbe/scratch/aaltonorms/results/zero_shot
LOG_FILE=$OUTPUT_PATH/$(printf 'results.out' $SLURM_ARRAY_TASK_ID)


# On triton, uncomment this to load the Python environment. On taito, you
# presumably installed Python yourself.
module load anaconda3


# Run the analyses


srun -o $LOG_FILE python zero_shot_decoding_leave1out.py aaltoprod cslb  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py aaltoprod vinson 
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py aaltoprod w2v_eng 
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py aaltoprod w2v_fin 
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py cslb aaltoprod  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py cslb vinson 
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py cslb w2v_eng 
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py cslb w2v_fin  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py vinson aaltoprod  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py vinson cslb  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py vinson w2v_eng 
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py vinson w2v_fin  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py w2v_eng aaltoprod  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py w2v_eng cslb  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py w2v_eng vinson  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py w2v_eng w2v_fin  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py w2v_fin aaltoprod  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py w2v_fin cslb  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py w2v_fin vinson  
srun -o $LOG_FILE python zero_shot_decoding_leave1out.py w2v_fin w2v_eng  
