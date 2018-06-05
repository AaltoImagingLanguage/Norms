# -*- coding: utf-8 -*-
"""
Generate norms DataArrays for comparisons and visualizations. 

@author: kivisas1
"""

import pandas
from scipy.io import loadmat

import numpy as np


#These are the files for different norm sets
stimulus_list = '/m/nbe/scratch/aaltonorms/stimuli/aaltonorms_stimulus_set.csv'
cslb_norms_file  = '/m/nbe/scratch/aaltonorms/CSLB_norms/feature_matrix.dat'
aalto_norms_file = '/m/nbe/scratch/aaltonorms/AaltoNorms/lemma_sorted20151027_dl_synmerge.mat'
vinson_norms_file = '/m/nbe/scratch/aaltonorms/VinsonFeatures/VinsonFeature_Matrix_edit.csv'
aalto85_norms_file = '/m/nbe/scratch/aaltonorms/Aalto85questions/Aalto85_sorted20160204.mat'
ginter_norms_file = '/m/nbe/scratch/aaltonorms/ginter/AaltoNorm_words/lemma/context_5+5/ginter_lemma_5+5/concepts_vectors.csv'
cmu_norms_file = '/m/nbe/scratch/aaltonorms/CMU_norms/bagOfFeatures.mat'


#Output data file names
cslb_out_file = '/m/nbe/scratch/aaltonorms/norms/cslb_norms_aalto_overlap.pkl'
vinson_out_file = '/m/nbe/scratch/aaltonorms/norms/vinson_norms_aalto_overlap.pkl'
aalto85_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto85_norms_aalto_overlap.pkl'
ginter_out_file = '/m/nbe/scratch/aaltonorms/norms/ginter_norms_aalto_overlap.pkl'
cmu_out_file = '/m/nbe/scratch/aaltonorms/norms/cmu_norms_aalto_overlap.pkl'
aalto_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_all.pkl'
aaltonorms_cslb_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_cslb_overlap.pkl'
aaltonorms_vinson_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_vinson_overlap.pkl'
aaltonorms_aalto85_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_aalto85_overlap.pkl'
aaltonorms_cmu_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_cmu_overlap.pkl'

#Get list of concept names
cslb_names = aaltostimuli["cslb"]
vinson_names = aaltostimuli["vinson"]
orignames = aaltostimuli["concept_fin"]
ginter_names = aaltostimuli["ginter"]
cmu_names = aaltostimuli["cmu"]

#Extract aalto production norm data from a Matlab file
def get_matlab_arrays(norm_file):
    fname =norm_file
    m = loadmat(fname)
    vectorarray = pandas.DataFrame(m['sorted']['mat'][0][0])
    wordarray = m['sorted']['word'][0][0]
    return vectorarray, wordarray


#Make aaltonorms dataframe with only those concepts that exist in the new norm
#set 
def select_aaltodata(names):
    df = pandas.DataFrame()
    for i, name in enumerate(names):
        if isinstance(names[i],unicode):
            origname = orignames[i] #Get corresponding aalto name
            df.name = origname
            data = aaltonorms.loc[[origname]] #Get corresponding row from aaltonorms
            df = df.append(data)
    return df

#Select concepts from a new norm set that exist in the aaltonorms 
def select_normdata(names, new_norms):
    df = pandas.DataFrame(columns=new_norms.columns.values)
    for i, name in enumerate(names):
        if isinstance(names[i],unicode):
            data = new_norms.loc[[name]]
            df = df.append(data)   
    return df


def select_normdata_notAalto (norm1, norm2):
    df1 = pandas.DataFrame()
    df2 = pandas.DataFrame()
    for i, name1 in enumerate(norm1.index.values):
        for j, name2 in enumerate(norm2.index.values):
            if name1 == name2:
                df1.name = name1
                df1.append()
                df2.name = name2
                df2 = df2.append(norm2.values[j])