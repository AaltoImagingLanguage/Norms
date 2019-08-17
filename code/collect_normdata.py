# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Wed Feb 13 12:11:37 2019

@author: saranpa1
"""

"""
Created on Fri Aug 24 13:55:53 2018

@author: kivisas1 & annika
"""


import pandas as pd
import os
from scipy.io import loadmat
import csv
import numpy as np

def get_matlab_arrays(norm_file):
    featurearray = []
    fname =norm_file
    m = loadmat(fname, variable_names='sorted')
    vectorarray = pd.DataFrame(m['sorted']['mat'][0][0])
    featurearray = m['sorted']['features'][0][0]
    featurearray = [s[0][0] for s in featurearray]
    wordarray = m['sorted']['word'][0][0]
    wordarray = [s[0][0] for s in wordarray]
    return vectorarray,featurearray, wordarray

def write_array2csv(outfile, array):
    with open(outfile, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for r in array: wr.writerow([r]) 
            
# LUT = look-up-table
#LUT_file = '/m/nbe/project/aaltonorms/data/concept_LUT.csv' # FIXME use the xls sheet instead
LUT_file = '/m/nbe/project/aaltonorms/data/SuperNormList.xls'
norms_path = '/m/nbe/project/aaltonorms/raw/'
out_path = '/m/nbe/project/aaltonorms/data/'
norms = [#'cslb', 
         #'vinson',
         #'aaltoprod', 
         #'aalto85',  
         #'cmu', 
         'mcrae']
# not implemented yet w2vFin and w2vEng  and 'mcrae
# w2vFin: 'Ginter/ginter-300-5+5/AaltoNorm_words/lemma/context_5+5/ginter_lemma_5+5/concepts_vectors.csv',

infiles = [#'CSLB/feature_matrix.dat', 
           #'Vinson/Vinson-BRM-2008/Vinson_feature_matrix_all.csv',
           #'AaltoProduction/concept_feature_matrix_pf0.1.csv',
           #'Aalto85questions/Aalto85_sorted20160204.mat',
           #'CMU/bagOfFeatures_inStruct.mat',
           'McRae/concept_feature_matrix.csv']

norms_dict = {}
for i in range(len(norms)):
    norms_dict[norms[i]] = infiles[i]
    
LUT = pd.read_excel(LUT_file, sheet_name=0, header=0, index_col=0) 



#Make lists of available concepts for each norm set (including overlapping words)
for norm in norms:
    print('Now running: ' + norm)
    data = LUT.dropna(subset=[norm]) #Select concepts according to whether it is available in each respective norm set. 
    if not os.path.exists(out_path + norm): 
       os.makedirs(out_path + norm)
    #Save as text file
    data.to_csv(out_path + norm + '/correspondence.csv', header=True, index=True,  sep='\t', encoding='utf-8')   


    if norms_dict.get(norm)[-3:]=='mat': 
        [temp_vectors, featurearray, wordarray] = get_matlab_arrays(norms_path + norms_dict.get(norm))
        write_array2csv(out_path + norm + '/features.csv', featurearray)
        wordarray = np.core.defchararray.lower(wordarray)
        write_array2csv(out_path + norm + '/vocab.csv', wordarray)
        
    else:   
        if  norm == 'vinson':             
            orig = pd.read_csv(norms_path + norms_dict.get(norm), header=0, 
                                      index_col=0, delimiter=',')
            orig = orig.transpose()
            
        elif  norm == 'cslb':  
            orig = pd.read_table(norms_path + norms_dict.get(norm), header=0, 
                              index_col=0)
        elif  norm == 'aaltoprod' or norm == 'mcrae':  
            orig = pd.read_csv(norms_path + norms_dict.get(norm), header=0, 
                              index_col=0, sep='\t')  
            #Remove one instance of metri, since this is a duplicate + "jolla",
            #which was ambiguous
            orig = orig.loc[~orig.index.duplicated(keep='first')]
            #orig = orig.drop("jolla")
  
    
        #Getvectors
        normwords = LUT[norm].dropna() 
        vectors = orig.loc[normwords] 
        # Get features 
        features = pd.DataFrame(orig.columns.values)        
        features.to_csv(out_path + norm + '/features.csv', header=False, index=False, sep='\t', encoding='utf-8')
        normwords.to_csv(out_path + norm + '/vocab.csv', header=False, index=False,  sep='\t', encoding='utf-8')
        
        # Save the remaining variable
        vectors.to_csv(out_path + norm + '/vectors.csv', header=False, index=False,  sep='\t', encoding='utf-8')



