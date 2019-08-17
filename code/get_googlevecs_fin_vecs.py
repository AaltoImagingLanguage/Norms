#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:46:23 2018

@author: kivisas1
"""
import sys
sys.path.append('/u/76/kivisas1/unix/semflu')
import gensim
import pandas as pd


#Get list of norm words
LUT = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', 
                    encoding='utf-8', 
                    header=0, index_col=0)

#Drop homonyms+verbs
LUT = LUT[LUT['homonym_verb']==0]
outdir = '/m/nbe/project/aaltonorms/data/w2v_fin/'
vocab = LUT['w2v_fin'].dropna()
w2v_file = '/m/nbe/project/aaltonorms/raw/w2v_fin/finnish_parsebank_v4_lemma_5+5.bin'

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)  


not_found = []
names = pd.DataFrame()
vectors = pd.DataFrame()
for index, word in enumerate(vocab):
    try:

       #Find w2v vector from the model and append the target word
       vector = model.get_vector(word)
       #Append vector + word to new data frame
       vectors = vectors.append(pd.Series(data=vector), ignore_index=True)
       #Save name of the vector
       names = names.append(pd.Series(word), ignore_index=True)
       
    except:
        not_found.append(word)    
        print(word + ' not in corpus')


names.to_csv(outdir + 'vocab.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')

vectors.to_csv(outdir + 'vectors.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')

not_found = pd.DataFrame(not_found)
not_found.to_csv(outdir + 'fin_google_concepts_not_found.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')

