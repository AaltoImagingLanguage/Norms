#!/bin/bash

# For testing, comment for main analysis
# SLURM_ARRAY_TASK_ID=5

# Make sure to request only the resources you really need to avoid cueing
#SBATCH --time=0:05:00
#SBATCH --mem-per-cpu=2G
#SBATCH -n 1


# Construct the names of the output files of this run
OUTPUT_PATH=/m/nbe/scratch/aaltonorms/results/zero_shot/perm/
LOG_FILE=$OUTPUT_PATH/$(printf 's%02d_results.out' $SLURM_ARRAY_TASK_ID)


# On triton, uncomment this to load the Python environment. On taito, you
# presumably installed Python yourself.
module load anaconda3


# Run the analyses


srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}aaltoprod_cslb/*_reg_results.mat -o ${OUTPUT_PATH}aaltoprod_cslb/results.csv -v
srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}aaltoprod_vinson/*_reg_results.mat -o ${OUTPUT_PATH}aaltoprod_vinson/results.csv -v
srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}aaltoprod_w2v_eng/*_reg_results.mat -o ${OUTPUT_PATH}aaltoprod_w2v_eng/results.csv -v
srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}aaltoprod_w2v_fin/*_reg_results.mat -o ${OUTPUT_PATH}aaltoprod_w2v_fin/results.csv -v
srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}cslb_aaltoprod/*_reg_results.mat -o ${OUTPUT_PATH}cslb_aaltoprod/results.csv -v
srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}cslb_vinson/*_reg_results.mat -o ${OUTPUT_PATH}cslb_vinson/results.csv -v
srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}cslb_w2v_eng/*_reg_results.mat -o ${OUTPUT_PATH}cslb_w2v_eng/results.csv -v
srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}cslb_w2v_fin/*_reg_results.mat -o ${OUTPUT_PATH}cslb_w2v_fin/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}vinson_aaltoprod/*_reg_results.mat -o ${OUTPUT_PATH}vinson_aaltoprod/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}vinson_cslb/*_reg_results.mat -o ${OUTPUT_PATH}vinson_cslb/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}vinson_w2v_eng/*_reg_results.mat -o ${OUTPUT_PATH}vinson_w2v_eng/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}vinson_w2v_fin/*_reg_results.mat -o ${OUTPUT_PATH}vinson_w2v_fin/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}w2v_eng_aaltoprod/*_reg_results.mat -o ${OUTPUT_PATH}w2v_eng_aaltoprod/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}w2v_eng_cslb/*_reg_results.mat -o ${OUTPUT_PATH}w2v_eng_cslb/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}w2v_eng_vinson/*_reg_results.mat -o ${OUTPUT_PATH}w2v_eng_vinson/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}w2v_eng_w2v_fin/*_reg_results.mat -o ${OUTPUT_PATH}w2v_eng_w2v_fin/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}w2v_fin_aaltoprod/*_reg_results.mat -o ${OUTPUT_PATH}w2v_fin_aaltoprod/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}w2v_fin_cslb/*_reg_results.mat -o ${OUTPUT_PATH}w2v_fin_cslb/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}w2v_fin_vinson/*_reg_results.mat -o ${OUTPUT_PATH}w2v_fin_vinson/results.csv -v
#srun  -o $LOG_FILE python analyze_results_leave1out.py ${OUTPUT_PATH}w2v_fin_w2v_eng/*_reg_results.mat -o ${OUTPUT_PATH}w2v_fin_w2v_eng/results.csv -v
