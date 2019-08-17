#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:59:22 2019

@author: kivisas1
"""

import pandas as pd
datadir = '/m/nbe/project/aaltonorms/raw/McRae/'
data = pd.read_csv(datadir + 'CONCS_FEATS_concstats_brm.txt', 
                   delimiter = "\t", index_col=0)
features=data['Feature'].unique()
concepts = data.index.unique()

#Initialize concept-feature matrix
feature_matrix = pd.DataFrame(0, columns=features, index=concepts)
for concept in concepts:
    short_list = data.loc[concept][['Feature','Prod_Freq']]
    for i, feat in enumerate(short_list['Feature']):
        #Drop duplicate features (if any)
        short_list.drop_duplicates(subset="Feature", keep="first", inplace=True)
        #Get production frequency for each feature
        PF = short_list[short_list['Feature']==feat]['Prod_Freq'][0]
        #Place PF to feature_matrix
        feature_matrix.loc[concept, feat] = PF 

feature_matrix.to_csv(datadir + "concept_feature_matrix.csv", sep = "\t")