#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:34:27 2018

@author: kivisas1
"""

"""
Learn a mapping from one norm dataset to another and evaluate its fit.

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
Sasa Kivisaari
"""
import os.path
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy.spatial import distance
#from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Learn a mapping from one norm dataset to another')
parser.add_argument('norm1', type=str, help='Norm data set 1')
parser.add_argument('norm2', type=str, help='Norm data set 2')
parser.add_argument('--reg', action='store_true', help='Whether to use regularization')
args = parser.parse_args()

print('Learning mapping from', os.path.basename(args.norm1), 'to', os.path.basename(args.norm2))
norm1 = args.norm1
norm2 = args.norm2
normpath = '/m/nbe/project/aaltonorms/data/'


#Get norm data 
LUT = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', encoding='utf-8', 
                       header=0, index_col=0)
norms1_vocab = pd.read_csv(normpath + norm1 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norms2_vocab = pd.read_csv(normpath + norm2 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norms1_vecs = pd.read_csv(normpath + norm1 + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', 
                         header=None, index_col=None)

norms2_vecs = pd.read_csv(normpath + norm2 + '/' + 'vectors.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=None)

#Choose words that are shared in the two norms
picks = LUT[LUT[norm1].notnull() & LUT[norm2].notnull()]
picks = picks.sort_values(by=["category"])

#Set word label as index
norms1_vecs.set_index(norms1_vocab.index, inplace=True)
norms2_vecs.set_index(norms2_vocab.index, inplace=True)

norm1 = norms1_vecs.loc[picks[norm1]].values
norm2 = norms2_vecs.loc[picks[norm2]].values

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

fig = plt.figure(figsize=(8, 8))
plt.imshow(confusion_matrix, cmap='gray_r', interpolation='nearest')
plt.xlabel('Which word I thought it was')
plt.ylabel('Which word it should have been')
plt.title('Confusion matrix')