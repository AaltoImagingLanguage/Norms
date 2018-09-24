#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:46:23 2018

@author: kivisas1
"""
import sys
sys.path.append('/u/76/kivisas1/unix/semflu')
import gensim
import gzip 
import logging
import pandas as pd
import numpy as np
#from visualize_distances import visualize_distances

#This is the Finnish model /m/nbe/project/corpora/ginter_downloaded_internet_models/fin_wform.bin


corpuspath = '/m/nbe/project/corpora/englishmodels'
fname = '/m/nbe/project/aaltonorms/raw/google_eng/eng_name_orig.csv'
root = '/m/nbe/project/aaltonorms/'
outdir = root + 'data/google_eng/'
#figdir = root + 'figs/'
fluency_eng = pd.read_table(fname, encoding='utf-8', header=None)
# Logging code taken from http://rare-technologies.com/word2vec-tutorial/
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Load Google's pre-trained Word2Vec model.
w2v_file = gzip.open(corpuspath +'/GoogleNews-vectors-negative300.bin.gz', mode='rb')
model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)  

# Retrieve the entire list of "words" from the Google Word2Vec model, and write
# these out to text files so we can peruse them.
vocab = model.vocab.keys()

fileNum = 1


not_found = []
names = []
vectors = pd.DataFrame()
for index, word in fluency_eng.iterrows():
    try:
       #Find w2v vector from the model and append the target word
       vector = np.append(word[0], model.get_vector(word[0]))
       #Append vector + word to new data frame
       vectors = vectors.append(pd.Series(data=vector), ignore_index=True)
       #Save name of the vector
       names.append(word[0])


    except:
        not_found.append(word[0])    
        print(word[0] + ' not in corpus')

vectors.to_csv(outdir + 'eng_google_words+vectors.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')

not_found = pd.DataFrame(not_found)
not_found.to_csv(outdir + 'eng_google_concepts_not_found.csv', header=False, index=False, 
                sep='\t', encoding='utf-8')

semflu_embeddings = vectors.iloc[:, 1:301].transpose()
#visualize_distances(semflu_embeddings, figdir + 'RDM_semflu_google_news_eng.pdf')
