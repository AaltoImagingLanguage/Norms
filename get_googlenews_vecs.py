#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:46:23 2018

@author: kivisas1, modified by @annika
"""
import sys
sys.path.append('/u/76/kivisas1/unix/semflu')
import gensim
import gzip 
import csv
import logging
import pandas as pd
import numpy as np

def write_array2csv(outfile, array, datatype):
    with open(outfile, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter='\t')
        if datatype == 'text':
            for r in array: wr.writerow([r]) 
        elif datatype == 'num':
            wr.writerows(array)
        else:
            print('specify datatype correctly')
            

# Logging code taken from http://rare-technologies.com/word2vec-tutorial/
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dpath = '/m/nbe/project/aaltonorms/'
 
# Load Google's pre-trained Word2Vec model
w2v_file = 'raw/w2v_eng/GoogleNews-vectors-negative300.bin.gz'
w2v_file = gzip.open(dpath + w2v_file, mode='rb')
model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)  

# Load the look-up table and select the correct column
LUT_file = 'data/SuperNormList.xlsx'
LUT = pd.read_excel(dpath +LUT_file, sheet_name=0, header=0, index_col=0)
data = LUT.dropna(subset=['w2v_eng']) 
data.to_csv(dpath + 'data/w2v_eng/correspondence.csv', header=True, index=True,  sep='\t', encoding='utf-8')   

# Retrieve the entire list of "words" from the Google Word2Vec model, and write
# these out to text files so we can peruse them.
vocab = model.vocab.keys()

not_found = []
names = []
vectors = []
for index, word in data.iteritems():
    try:
       #Find w2v vector from the model and append the target word
       vector = np.append(word, model.get_vector(word))
       #Append vector + word to new data frame
       vectors.append(model.get_vector(word))
       #Save name of the vector
       names.append(word)
    except:
        not_found.append(word)    
        print(word + ' not in corpus')

#Save the results
write_array2csv(dpath + 'data/w2v_eng/vectors.csv', vectors, 'num')
not_found = pd.DataFrame(not_found)
not_found.to_csv(dpath + 'data/w2v_eng/vocab_not_found.csv', header=False, index=False, sep='\t', encoding='utf-8')
write_array2csv(dpath + 'data/w2v_eng/vocab.csv', names, 'text')

