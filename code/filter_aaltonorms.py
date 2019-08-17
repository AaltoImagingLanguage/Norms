#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
#import numpy as np

outpath = "/m/nbe/project/aaltonorms/raw/AaltoProduction/"
dp_path = '/m/nbe/project/aaltonorms/raw/AaltoProduction/dep_parsed/'

filenames = [dp_path + 'normi.%s.cleank.lemmafreq.csv' % i for i in range(1,301)]



targets = pd.read_csv('/m/nbe/project/aaltonorms/raw/AaltoProduction/raw_data/normnames_corrected.txt', 
                      encoding='utf-8', header=None, names=['target']) 

production_freqs = pd.read_csv('/m/nbe/project/aaltonorms/raw/AaltoProduction/raw_data/numRespond.txt', 
                               header = None, delimiter="\t", index_col=0, 
                               names = ["target", "pf"])

#features_long = pd.read_csv('/m/nbe/project/aaltonorms/raw/AaltoProduction/features_LUT.csv',
#                       delimiter = "\t", index_col=None,header=None)

features = pd.read_csv('/m/nbe/project/aaltonorms/raw/AaltoProduction/features_LUT.csv',
                       index_col=None,header=None)

features = features.apply(lambda x: x.str.strip())

#Exclude stopwords
stopwords= pd.read_csv('/m/nbe/project/aaltonorms/raw/AaltoProduction/omorfiLemmaDependencyParsed/AaltoProd.stopwords_slk.txt', 
                       encoding='utf-8', header=None)[0].tolist()

#Initialize full conceptxfeature matrix
full_matrix = pd.DataFrame(0, index=targets['target'], columns = list(range(0, len(features))))

for i, file in enumerate(filenames):
    target = targets.iloc[i][0]
    n = i+1
    freqs = pd.read_csv(filenames[i], encoding='utf-8', header=None, 
                           names=['Freq', 'Word'])
    #Get rid of nans
    freqs = freqs.dropna(axis=0, how='any')
    
    #Exclude stopwords
    freqs =  freqs[~freqs['Word'].isin(stopwords)]
    
    #Normalize by number of respondents
    pf = production_freqs["pf"].iloc[0] 
    freqs['Freq'] = freqs['Freq']/pf
    
    #Excelude features produced by > 10% respondents
    freqs = freqs[freqs['Freq']>=0.1]
    
    #Write to file
    #freqs.to_csv(outpath + 'lemmas_pf>3/normi.' + str(n) + '.filt_norm.lemmafreq.csv')
    
    #Find synonyms for each feature
    
    for j, word in enumerate(freqs['Word']):
        #Get rid of leading spaces
        word = word.strip()
        
        #First look for the whole string
        if features.isin([word]).any(axis=1).any(axis=0):
            ind = features.index[features.isin([word]).any(axis=1)].tolist()[0]
            full_matrix.loc[target, ind] = freqs['Freq'].iloc[j]

        #Check if this is a compond word
        else:
            if "#" in word:
                #Try removing the hash
                word2 = word.replace("#","")
                if features.isin([word2]).any(axis=1).any(axis=0):
                    #Get feature index
                    ind = features.index[features.isin([word2]).any(axis=1)].tolist()[0]
                    full_matrix.loc[target, ind] = freqs['Freq'].iloc[j]
                else:
                    #Try substrings separately 
                    items = word.split("#")                        
                    for item in items:
                          if features.isin([item]).any(axis=1).any(axis=0):
                               #Get feature index
                                ind = features.index[features.isin([item]).any(axis=1)].tolist()[0]
                                #Replace matrix value by weighted frequency
                                full_matrix.loc[target, ind] = freqs['Freq'].iloc[j]
                          else:
                              print(word + " or " + item + " is not listed as a feature in LUT")
            else:
                print(word + " is not listed as a feature in LUT")

#Trim feature set to include only such features that exist in data (using the)
#selected pf threshold

zerocols = (full_matrix != 0).any(axis=0)
trimmed_matrix = full_matrix.loc[:, zerocols]
trimmed_features = features.loc[zerocols,:]

#Change column names
trimmed_matrix.columns = trimmed_features[0]      
trimmed_matrix.to_csv(outpath + "concept_feature_matrix_pf0.1.csv", sep = "\t")          
trimmed_features.to_csv(outpath + "featurelist_pf0.1.csv", sep = "\t") 