#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:40:27 2019

@author: kivisas1
"""
from scipy.io import loadmat
import itertools
from matplotlib import pyplot as plt

path = "/m/nbe/project/aaltonorms/results/zero_shot/"
figure_dir = "/m/nbe/project/aaltonorms/figures/"
norms = ["aaltoprod", "cslb", "vinson", "w2v_eng", "w2v_fin"]
conf_mat = zeros([98,98])
for pair in list(itertools.permutations(norms, 2)):
    mfile = loadmat(path + pair[0] + "_" + pair[1] + "_reg_results.mat")
    conf_mat = conf_mat + mfile['confusion_matrix']
    
    

fig = plt.figure(figsize=(8, 8))
plt.imshow(conf_mat, interpolation='nearest', cmap ="plasma")
ax = plt.gca()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_xaxis().set_ticklabels([])
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_yaxis().set_ticklabels([])
#plt.xlabel('Correct concept')
#plt.ylabel('Predicted concept')
#plt.title('Confusion matrix')
plt.colorbar(fraction=0.046, pad=0.04)
plt.savefig(figure_dir + "confusion_matrix.png", format='png', dpi=1000, 
            bbox_inches='tight')

