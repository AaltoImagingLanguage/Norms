#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:56:37 2019

@author: kivisas1
"""


import pandas as pd 

#outpath = '/m/nbe/scratch/aaltonorms/results/zero_shot/perm'
normpath = '/m/nbe/scratch/aaltonorms/data/'
norms = ["aaltoprod", "cslb", "vinson", "w2v_eng", "w2v_fin"]


#Get data from the big excel file
LUT = pd.read_excel('/m/nbe/scratch/aaltonorms/data/SuperNormList.xls', 
                    encoding='utf-8', 
                    header=0, index_col=0)

#Exclude homonyms, verbs and abstract words
LUT = LUT[LUT['action_words']==0]
LUT = LUT[LUT['category']!="abstract_mid"]
LUT = LUT[LUT['category']!="abstract_high"]

picks = LUT[LUT[norms[0]].notnull() & LUT[norms[1]].notnull() &
            LUT[norms[2]].notnull() & LUT[norms[3]].notnull() & 
            LUT[norms[4]].notnull()]
picks = picks.sort_values(by=["category"])


catinds = {
		"animal" : 1,
		"bodypart" : 2,
		"building" : 3,
		"clothing" : 4,
		"container" : 5,
		"fruit" : 6,
        "furniture" : 7,
        "tool": 8,
        "vegetable" : 9,
        "vehicle" : 10,
        "weapon" : 11
        
	}

item_index = picks[['eng_name', 'category']]
item_index  = item_index.replace({"category": catinds})
item_index.to_csv(path_or_buf = "/m/nbe/scratch/aaltonorms/data/item_category_list.csv", 
                  index = False, header=False, sep="\t")
item_index.to_csv(path_or_buf = "/m/nbe/project/aaltonorms/data/item_category_list.csv", 
                  index = False, header=False, sep="\t")