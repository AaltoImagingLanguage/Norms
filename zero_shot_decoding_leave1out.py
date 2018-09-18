#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:34:27 2018

@author: kivisas1
"""

"""
Learn a mapping from one norm dataset to another and evaluate its fit.

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import os.path
import argparse
import pandas
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy.spatial import distance
#from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Learn a mapping from one norm dataset to another')
parser.add_argument('norm1', type=str, help='Filename for norm data set 1 (must end in .csv)')
parser.add_argument('norm2', type=str, help='Filename for norm data set 2 (must end in .csv)')
parser.add_argument('--reg', action='store_true', help='Whether to use regularization')
args = parser.parse_args()

print('Learning mapping from', os.path.basename(args.norm1), 'to', os.path.basename(args.norm2))
#norm1 = loadmat(args.norm1)['newVectors']
#norm2 = loadmat(args.norm2)['newVectors']

norm1 = pandas.read_table(args.norm1, encoding='utf-8', header=None,index_col=0).values
norm2 = pandas.read_table(args.norm2, encoding='utf-8', header=None,index_col=0).values

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