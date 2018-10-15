#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 13:55:53 2018

@author: kivisas1
"""
import pandas as pd
import os
from scipy.io import loadmat

LUT_file = '/m/nbe/project/aaltonorms/data/concept_LUT.csv'
norms_path = '/m/nbe/project/aaltonorms/raw/'
out_path = '/m/nbe/project/aaltonorms/data/'
norms = ['cslb', 'aaltoprod', 'aalto85', 'vinson', 'cmu', 'mcrae']

LUT = pd.read_table(
    LUT_file, encoding='utf-8', header=0, index_col=0,
)

#duplicates = LUT[LUT.duplicated(['FIN_name'], keep=False)]


#These are the files for different norm sets
cslb_norms_file  = norms_path + 'CSLB/feature_matrix.dat'
aalto300_file = norms_path + 'AaltoProduction/lemma_sorted20151027_dl_synmerge.mat'
vinson_norms_file = norms_path + 'Vinson/VinsonFeature_Matrix.csv'
aalto85_norms_file = norms_path + 'Aalto85questions/Aalto85_sorted20160204.mat'
ginter_norms_file = norms_path + 'Ginter/ginter-300-5+5/AaltoNorm_words/lemma/context_5+5/ginter_lemma_5+5/concepts_vectors.csv'
cmu_norms_file = norms_path + 'CMU/bagOfFeatures.mat'


#Make lists of available concepts for each norm set (including overlapping words)
for norm in norms:
    data = LUT.dropna(subset=[norm]) #Select concepts according to whether it is available in each respective norm set. 
    if not os.path.exists(out_path + norm): 
       os.makedirs(out_path + norm)
    #Save as text file
    data.to_csv(out_path + norm + '/correspondence.csv', header=True, index=True, 
                sep='\t', encoding='utf-8')   



#Get feature vectors
#This is the CSLB semantic concept x features matrix
#Dimensions: 638 (concepts) x 2725(semantic features) array
    
cslb = LUT['cslb'].dropna() #List of CSLB stimuli
cslb_orig = pd.read_table(cslb_norms_file, encoding='latin1', 
                                   header=0, index_col=0)    
cslb_vectors = cslb_orig.loc[cslb]

#Save feature list to file
cslb_features = pd.DataFrame(cslb_orig.columns.values)
cslb_features.to_csv(out_path + 'cslb' + '/features.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')
cslb.to_csv(out_path + 'cslb' + '/vocab.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')
cslb_vectors.to_csv(out_path + 'cslb' + '/vectors.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')

#Vinson full semantic concept x features matrix
#Dimensions: 173 (concepts) x 1027(semantic features) array
vinson = LUT['vinson'].dropna() 
vinson_orig = pd.read_table(vinson_norms_file, encoding='latin1', 
                                     header=0, index_col=0, delimiter=",")
vinson_orig = vinson_orig.transpose()
vinson_vectors = vinson_orig.loc[vinson]

#Save feature list to file
vinson_features = pd.DataFrame(vinson_orig.columns.values)
vinson_features.to_csv(out_path + 'vinson' + '/features.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')
#Save list of concepts
vinson.to_csv(out_path + 'vinson' + '/vocab.csv', header=False, index=False)
#Save list of vectors
vinson_vectors.to_csv(out_path + 'vinson' + '/vectors.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')

norm_file = aalto300_file


def get_matlab_arrays(norm_file):
    fname =norm_file
    m = loadmat(fname)
    vectorarray = pd.DataFrame(m['sorted']['mat'][0][0])
    wordarray = m['sorted']['word'][0][0]
    return vectorarray, wordarray
 
 [x, y] = get_matlab_arrays(aalto300_file)