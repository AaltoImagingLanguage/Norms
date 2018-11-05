#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:46:23 2018

@author: @annika
"""
#import sys
#sys.path.append('/u/76/kivisas1/unix/semflu')
import gensim
import logging
import pandas as pd
import numpy as np
import csv

def write_array2csv(outfile, array, datatype):
    with open(outfile, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter='\t')
        if datatype == 'text':
            for r in array: wr.writerow([r]) 
        elif datatype == 'num':
            wr.writerows(array)
        else:
            print('specify datatype correctly')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dpath = '/m/nbe/project/aaltonorms/'
   
# corpora with lingsoft parsing: 
w2v_file = 'raw/w2v_fin/finnish_parsebank_v4_lemma_5+5.bin'

# Load the look-up table and select the correct column
LUT_file = 'data/SuperNormList.xlsx'
LUT = pd.read_excel(dpath +LUT_file, sheet_name=0, header=0, index_col=0)
data = LUT['w2v_fin'].dropna() 

# Load the pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format(dpath + w2v_file, binary=True)  

# Retrieve the entire list of "words" from the Word2Vec model, and write
# these out to text files so we can peruse them.
vocab = model.vocab.keys()

not_found = []
names = []
vectors = []
#pd.DataFrame()
for index, word in data.iteritems():
    try:
       #Find w2v vector from the model and append the target word
       vector = np.append(word, model.get_vector(word))
       #Append vector + word to new data frame
       vectors.append(model.get_vector(word))
       #append the target word of the vector
       names.append(word)
    except:
        not_found.append(word)    
        print(word + ' not in corpus')

#Save the results
write_array2csv(dpath + 'data/w2v_fin/vectors.csv', vectors, 'num')        
not_found = pd.DataFrame(not_found)
not_found.to_csv(dpath + 'data/w2v_fin/vocab_not_found.csv', header=False, index=False, sep='\t', encoding='utf-8')
write_array2csv(dpath + 'data/w2v_fin/vocab.csv', names, 'text')

