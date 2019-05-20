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
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy.spatial import distance
from scipy.io import savemat


parser = argparse.ArgumentParser(description='Learn a mapping from one norm \
                                 dataset to another')
parser.add_argument('norm1', type=str, help='Norm data set 1')
parser.add_argument('norm2', type=str, help='Norm data set 2')
parser.add_argument('-i', '--iteration', metavar='N', type=int, default=1,
                    help='The iteration (as a number). This number is recorded in the output .mat file. Defaults to 1.')
parser.add_argument('-o', '--output', metavar='filename', type=str,
                    help='The file to write the results to; should end in .mat. Defaults to ./iteration_0001_results.mat.')
parser.add_argument('--reg', action='store_true', help='Whether to use \
                    regularization')
args = parser.parse_args()

print('Learning mapping from', os.path.basename(args.norm1), 'to', 
      os.path.basename(args.norm2))

outpath = '/m/nbe/project/aaltonorms/results/zero_shot/'
normpath = '/m/nbe/project/aaltonorms/data/'
norms = ["aaltoprod", "cslb", "vinson", "w2v_eng", "w2v_fin"]

if args.reg == None:
    output = outpath + args.norm1 + "_" + args.norm2 + "_results.mat"
else:
    output = outpath + args.norm1 + "_" + args.norm2 + "_reg_results.mat"
#Get data from the big excel file
LUT = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', 
                    encoding='utf-8', 
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
#Form permutation test, shuffle order of norm1
np.random.shuffle(norm1)

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
    'accuracy': accuracy,
    'iteration': args.iteration,
    'confusion_matrix': confusion_matrix
}

savemat(args.output, results)

