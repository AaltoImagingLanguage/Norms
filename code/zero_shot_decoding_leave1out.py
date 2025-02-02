#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:34:27 2018
Learn a mapping from one norm dataset to another and evaluate its fit.
Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
Sasa Kivisaari
"""
import os.path
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy.spatial import distance
from scipy.io import savemat

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


parser = argparse.ArgumentParser(description='Learn a mapping from one norm \
                                 dataset to another')
parser.add_argument('norm1', type=str, help='Norm data set 1')
parser.add_argument('norm2', type=str, help='Norm data set 2')
parser.add_argument('--reg', action='store_true', help='Whether to use \
                    regularization')
args = parser.parse_args()

print('Learning mapping from', os.path.basename(args.norm1), 'to',
      os.path.basename(args.norm2))

# outpath = '/m/nbe/scratch/aaltonorms/results/zero_shot/'
outpath = './results/'
normpath = '/m/nbe/scratch/aaltonorms/data/'
norms = ["aaltoprod", "cslb", "vinson", "w2v_eng", "w2v_fin"]
analysis = args.norm1 + "_" + args.norm2

os.makedirs(outpath + analysis, exist_ok=True)

#Determine output filename
if args.reg == None:
    output = outpath + analysis + "/leave1out_results.mat"
else:
    output = outpath + analysis + "/leave1out_reg_results.mat"

#Get data from the big excel file
LUT = pd.read_excel('/m/nbe/scratch/aaltonorms/data/SuperNormList.xls',
                    header=0, index_col=0)

#Exclude homonyms, verbs and abstract words
LUT = LUT[LUT['action_words']==0]
LUT = LUT[LUT['category']!="abstract_mid"]
LUT = LUT[LUT['category']!="abstract_high"]

norm1_vocab = pd.read_csv(normpath + args.norm1 + '/' + 'vocab.csv',
                          encoding='utf-8',
                          delimiter = '\t', header=None, index_col=0)

norm2_vocab = pd.read_csv(normpath + args.norm2 + '/' + 'vocab.csv',
                          encoding='utf-8',
                          delimiter = '\t', header=None, index_col=0)

norm1_vecs = pd.read_csv(normpath + args.norm1 + '/' + 'vectors.csv',
                         encoding='utf-8', delimiter = '\t',
                         header=None, index_col=None)

norm2_vecs = pd.read_csv(normpath + args.norm2 + '/' + 'vectors.csv',
                         encoding='utf-8',
                         delimiter = '\t', header=None, index_col=None)

picks = LUT[LUT[norms[0]].notnull() & LUT[norms[1]].notnull() &
            LUT[norms[2]].notnull() & LUT[norms[3]].notnull() &
            LUT[norms[4]].notnull()]

print("Number of words: " + str(len(picks)))
picks = picks.sort_values(by=["category"])

#Set word label as index
norm1_vecs.set_index(norm1_vocab.index, inplace=True)
norm2_vecs.set_index(norm2_vocab.index, inplace=True)

norm1_vecs = norm1_vecs.loc[picks[args.norm1]]
norm2_vecs = norm2_vecs.loc[picks[args.norm2]]

norm1 = norm1_vecs.values
norm2 = norm2_vecs.values

num_words = len(norm1)
assert len(norm2) == num_words

# Learn mapping from norm1 to norm2 in a leave-one-out setting
norm2_pred = np.zeros_like(norm2)

if args.reg:
    print('Using regularization')
    # Regularized Linear Regression
    mapping = RidgeCV(alphas=np.logspace(-5, 5, 100))
else:
    print('Not using regularization')
    # Plain Linear Regression
    mapping = LinearRegression()

for train, test in LeaveOneOut().split(norm1, norm2):
    mapping.fit(norm1[train], norm2[train])
    norm2_pred[test] = mapping.predict(norm1[test])

# See how well norm2_pred fits to norm2, in terms of accuracy
dist = distance.cdist(norm2_pred, norm2, metric='euclidean')
accuracy = np.mean(dist.argmin(axis=1) == np.arange(num_words))
print('Accuracy:', accuracy * 100, '%')

# Plot the confusion matrix
confusion_matrix = np.zeros_like(dist)
confusion_matrix[np.arange(num_words), dist.argmin(axis=1)] = 1

results = {
    'overall_accuracy': accuracy,
    'distance_matrix': dist,
    'confusion_matrix': confusion_matrix,
    'words': picks['eng_name'].values,
    'categories': picks['category'].values,
}

savemat(output, results)
print("Saved results to: " + output)


#Plot results
#fig = plt.figure(figsize=(8, 8))
#plt.imshow(confusion_matrix, cmap='gray_r', interpolation='nearest')
#plt.xlabel('Which word I thought it was')
#plt.ylabel('Which word it should have been')
#plt.title('Confusion matrix')
#fig.save(outpath + norm1 + "_" + norm2 + "_confusion_matrix.png" )
