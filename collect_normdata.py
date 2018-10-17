#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 13:55:53 2018

@author: kivisas1 & annika
"""
import pandas as pd
import os
from scipy.io import loadmat
import csv

def get_matlab_arrays(norm_file):
    featurearray = []
    fname =norm_file
    m = loadmat(fname, variable_names='sorted')
    vectorarray = pd.DataFrame(m['sorted']['mat'][0][0])
    featurearray = m['sorted']['features'][0][0]
    featurearray = [s[0][0] for s in featurearray]
    return vectorarray,featurearray

def write_array2csv(outfile, array):
    with open(outfile, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator='\n')
        for r in array: wr.writerow([r]) 
            
# LUT = look-up-table
LUT_file = '/m/nbe/project/aaltonorms/data/concept_LUT.csv'
norms_path = '/m/nbe/project/aaltonorms/raw/'
out_path = '/m/nbe/project/aaltonorms/data/'
norms = ['cslb', 'vinson','aaltoprod', 'aalto85',  'cmu']
# not implemented yet w2vFin and w2vEng  and 'mcrae
# w2vFin: 'Ginter/ginter-300-5+5/AaltoNorm_words/lemma/context_5+5/ginter_lemma_5+5/concepts_vectors.csv',

infiles = ['CSLB/feature_matrix.dat', 
           'Vinson/VinsonFeature_Matrix.csv',
           'AaltoProduction/lemma_sorted20151027_dl_synmerge.mat',
           'Aalto85questions/Aalto85_sorted20160204.mat',
           'CMU/bagOfFeatures_inStruct.mat']

norms_dict = {}
for i in range(len(norms)):
    norms_dict[norms[i]] = infiles[i]
    
    
LUT = pd.read_table(LUT_file, encoding='utf-8', header=0, index_col=0)
#duplicates = LUT[LUT.duplicated(['FIN_name'], keep=False)]

#Make lists of available concepts for each norm set (including overlapping words)
for norm in norms:
    print('Now running: ' + norm)
    data = LUT.dropna(subset=[norm]) #Select concepts according to whether it is available in each respective norm set. 
    if not os.path.exists(out_path + norm): 
       os.makedirs(out_path + norm)
    #Save as text file
    data.to_csv(out_path + norm + '/correspondence.csv', header=True, index=True,  sep='\t', encoding='utf-8')   

    #Select the words in a praticular norm set
    temp = LUT[norm].dropna()    

    if norms_dict.get(norm)[-3:]=='mat': 
        [temp_vectors, featurearray] = get_matlab_arrays(norms_path + norms_dict.get(norm))
        write_array2csv(out_path + norm + '/features.csv', featurearray)
        
    else:    
        if  norms_dict.get(norm)[-3:]=='csv':  
            delim = ','
        else: # data is dat
            delim = None
            
        temp_orig = pd.read_table(norms_path + norms_dict.get(norm), encoding='latin1', header=0, index_col=0, delimiter=delim)    
        
        if  norm == 'vinson': 
            temp_orig = temp_orig.transpose()        
        #Getvectors
        temp_vectors = temp_orig.loc[temp] 
        # Get features 
        #FIXME w2v doesn't have feature labels so this step needs to be optional
        temp_features = pd.DataFrame(temp_orig.columns.values)        
        temp_features.to_csv(out_path + norm + '/features.csv', header=False, index=False, sep='\t', encoding='utf-8')
        
    # Save the remaining variables
    temp_vectors.to_csv(out_path + norm + '/vectors.csv', header=False, index=False,  sep='\t', encoding='utf-8')
    temp.to_csv(out_path + norm + '/vocab.csv', header=False, index=False,  sep='\t', encoding='utf-8')

