#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:39:02 2019

@author: saranpa1
"""
### This code takes all the common words from all corpuses and compares them in pairs.

import pandas as pd
from scipy.spatial.distance import squareform, pdist
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import dendrogram, leaves_list
import argparse
import os

#import pylab
#import scipy.cluster.hierarchy as sch
#from sklearn.cluster import  AgglomerativeClustering



normpath = '/m/nbe/project/aaltonorms/data/'
#Distance metric
dist = "cosine"
#Directory for the figures to be created
figure_dir = '/m/nbe/project/aaltonorms/figures/'

aino_dir = '/u/85/saranpa1/unix/aaltonorms/figures/'

parser = argparse.ArgumentParser(description='Learn a mapping from one norm dataset to another')
parser.add_argument('norm1', type=str, help='Norm data set 1')
parser.add_argument('norm2', type=str, help='Norm data set 2')
parser.add_argument('norm3', type=str, help='Norm data set 3')
parser.add_argument('norm4', type=str, help='Norm data set 4')
parser.add_argument('norm5', type=str, help='Norm data set 5')
parser.add_argument('--reg', action='store_true', help='Whether to use regularization')
args = parser.parse_args()

print('Comparing', os.path.basename(args.norm1), 'to', os.path.basename(args.norm2), 'to', os.path.basename(args.norm3),
      'to', os.path.basename(args.norm4), 'to', os.path.basename(args.norm5))
norm1 = args.norm1
norm2 = args.norm2
norm3 = args.norm3
norm4 = args.norm4
norm5 = args.norm5


#Get data from the big excel file
LUT = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', encoding='utf-8', 
                       header=0, index_col=0)

#Exclude homonyms
LUT = LUT[LUT['action_words']==0]
LUT = LUT[LUT['category']!="abstract_mid"]
LUT = LUT[LUT['category']!="abstract_high"]


norm1_vocab = pd.read_csv(normpath + norm1 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norm2_vocab = pd.read_csv(normpath + norm2 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norm3_vocab = pd.read_csv(normpath + norm3 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norm4_vocab = pd.read_csv(normpath + norm4 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norm5_vocab = pd.read_csv(normpath + norm5 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norm1_vecs = pd.read_csv(normpath + norm1 + '/' + 'vectors.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=None)

norm2_vecs = pd.read_csv(normpath + norm2 + '/' + 'vectors.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=None)

norm3_vecs = pd.read_csv(normpath + norm3 + '/' + 'vectors.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=None)

norm4_vecs = pd.read_csv(normpath + norm4 + '/' + 'vectors.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=None)

norm5_vecs = pd.read_csv(normpath + norm5 + '/' + 'vectors.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=None)


picks = LUT[LUT[norm1].notnull() & LUT[norm2].notnull() & LUT[norm3].notnull() & LUT[norm4].notnull() & LUT[norm5].notnull()]
picks = picks.sort_values(by=["category"])

#Set word label as index
norm1_vecs.set_index(norm1_vocab.index, inplace=True)
norm2_vecs.set_index(norm2_vocab.index, inplace=True)
norm3_vecs.set_index(norm3_vocab.index, inplace=True)
norm4_vecs.set_index(norm4_vocab.index, inplace=True)
norm5_vecs.set_index(norm5_vocab.index, inplace=True)

norm1_vecs = norm1_vecs.loc[picks[norm1]]
norm2_vecs = norm2_vecs.loc[picks[norm2]]
norm3_vecs = norm3_vecs.loc[picks[norm3]]
norm4_vecs = norm4_vecs.loc[picks[norm4]]
norm5_vecs = norm5_vecs.loc[picks[norm5]]


#Check dimensions
assert len(norm1_vecs) == len(norm2_vecs) == len(norm3_vecs) == len(norm4_vecs) == len(norm5_vecs)
print("Number of overlapping words: " + str(len(norm1_vecs)))

### Single matrises with common words from all the matrises!

def get_distances(norms):
    distmat = squareform(pdist(norms, metric=dist))
    distmat_full = list(distmat)
    tri = np.tril_indices(norms.shape[0]) #For denoting lower triangular 
    distmat[tri] = np.nan   # comment this if you want to visualize the whole matrix (affects the numbers)
    distvector = np.asarray(distmat.reshape(-1)) #Take upper triangular and reshape
    distvector = distvector[~np.isnan(distvector)]
    return distmat_full, distvector


def remove_ticks(ax):
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticklabels([])
    return ax

def make_category_bar(cats):
    plt.figure(figsize=(1,3))    
    ax = plt.imshow(cats, cmap='Paired', interpolation='nearest', extent=[0,5,0,1], aspect=100)
    ax = remove_ticks(ax)
    return ax
    

def compare_norms(A,B, label_A, label_B,  cats=None):
    plt.figure(figsize=(10,3))  
    ax1 = plt.subplot(1,2,1)
    plt.title(label_A)
    ax1 = remove_ticks(ax1)
    plt.imshow(get_distances(A)[0], cmap="plasma", interpolation='nearest')
    #Y = sch.linkage(get_distances(A), method='centroid')
    #Z = sch.dendrogram(Y, orientation='right')
    #index = Z(A, method='complete', metric=dist) ['leaves']
    #A = get_distances(A)[index,:]
    #A = get_distances(A)[index,:]
    ax2 = plt.subplot(1,2,2)
    ax2 = remove_ticks(ax2)
    plt.title(label_B)
    plt.imshow(get_distances(B)[0], cmap="plasma", interpolation='nearest')
    plt.clim(0, 1);
    plt.colorbar(ax=[ax1, ax2])
    rho, pval = spearmanr(get_distances(A)[1], get_distances(B)[1])
   
    print(label_A + " vs. " + label_B + "  rho is: " + 
        str(round(rho,3)) + ", pvalue = " + str(round(pval,5)))
    plt.savefig(figure_dir + label_A + "_" + label_B + "_production_norm_comparison.pdf", 
                format='pdf', dpi=1000, bbox_inches='tight')


index = []
def hierarchical_clustering(norms,labels,font):
    Z = hac.linkage(norms, method='complete', metric=dist)   
    labels = labels
    # Plot dendogram
    plt.figure(figsize=(25, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    #plt.xlabel('Norm')
    plt.ylabel('Distance')
    hac.dendrogram(
        Z,
        labels=labels,
        leaf_rotation = 90.,  # rotates the x axis labels
        leaf_font_size= font,  # font size for the x axis labels
    )
    #
    plt.show()
    index = leaves_list(Z)
    return index,Z


norm1_index,norm1_Z = hierarchical_clustering(norm1_vecs.values, picks['eng_name'].tolist(),'8.')
#plt.savefig(figure_dir + norm1 + "_hierarchical_clustering.pdf", 
            #format='pdf', dpi=1000, bbox_inches='tight')
norm2_index,norm2_Z = hierarchical_clustering(norm2_vecs.values, picks['eng_name'].tolist(),'8.')
#plt.savefig(figure_dir + norm2 + "_hierarchical_clustering.pdf", 
            #format='pdf', dpi=1000, bbox_inches='tight')

compare_norms(norm1_vecs.values, norm2_vecs.values, norm1, norm2)

def get_different_clusters(Z):
    D1=dendrogram(Z,
                  color_threshold=1,
                  p=40,
                  truncate_mode='lastp',
                  distance_sort='ascending')
    plt.close()
    D2=dendrogram(Z,
                  #color_list=['g',]*7,
                  p=102,
                  truncate_mode='lastp',
                  distance_sort='ascending')
    plt.close()
    from itertools import groupby
    n = [list(group) for key, group in groupby(D2['ivl'],lambda x: x in D1['ivl'])]
    return n

