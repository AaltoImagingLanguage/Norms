 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:10:22 2018

@author: kivisas1
"""


import numpy as np 
import argparse
import pandas as pd



parser = argparse.ArgumentParser(description='Learn a mapping from one norm \
                                 dataset to another')
parser.add_argument('type', type=str, default = 'main', help='Type of analysis: \
                    main or permutation. Defaults to Main.')
parser.add_argument('-fname', type=str, default = 'reg_results.csv', help='Result \
                    filename. Defaults to reg_results.csv: \
                    main or permutation. Defaults to Main.')
args = parser.parse_args()

root_dir = '/m/nbe/scratch/aaltonorms/results/zero_shot/'


ananames = ['aaltoprod_cslb',  'aaltoprod_vinson',  'aaltoprod_w2v_eng',
            'aaltoprod_w2v_fin', 'cslb_aaltoprod',  'cslb_vinson',
            'cslb_w2v_eng', 'cslb_w2v_fin', 'vinson_aaltoprod', 'vinson_cslb',
            'vinson_w2v_eng', 'vinson_w2v_fin', 'w2v_eng_aaltoprod',
            'w2v_eng_cslb', 'w2v_eng_vinson', 'w2v_eng_w2v_fin',
            'w2v_fin_aaltoprod', 'w2v_fin_cslb', 'w2v_fin_vinson',
            'w2v_fin_w2v_eng']

percentile = 99.75
levels = ["item-level", "category-level"] 

if args.type == "permutation":

    for level in levels:
        print("\n##################################\n")
        print(level + ", Percentile: " + str(percentile) + "\n")
        for ananame in ananames:
            fname = root_dir + 'perm/' + ananame + '/' + args.fname
            results = pd.read_csv(fname, encoding='utf-8', sep=",", header=0)

            average_perm = np.average(results[level].values)
            perc_score = np.percentile(results[level].values, percentile)

            #print("Ananame: " + str(round(average_perm,3)))
            print(ananame + "\t" + str(round(perc_score*100,4)))
print("\n##################################\n")
        
        
if args.type == "main":
    for level in levels:
        print("\n##################################\n")
        print(level + "\n")
        for ananame in ananames:
            fname = root_dir + ananame + '/' + args.fname
            results = pd.read_csv(fname, encoding='utf-8', sep=",", header=0)
            summary = pd.DataFrame()

            acc =results[level].values[0]*100
            print(ananame + "\t" +  str(round(acc,4)))

#            

