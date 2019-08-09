"""
This script extracts different sets of semantic norms (Aalto, CSLB, Vinson, 
Corpus, Questionnaire) and compares them.

@author: kivisas1 (sasa@neuro.hut.fi)
Last update: 30.5.2018
"""

import pandas as pd
from scipy.spatial.distance import squareform, pdist
from matplotlib import pyplot as plt
import numpy as np
#from scipy.stats import spearmanr
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import dendrogram, leaves_list
#import argparse
#import os

normpath = '/m/nbe/project/aaltonorms/data/'
#Distance metric
dist = "cosine"
#Directory for the figures to be created
figure_dir = '/m/nbe/project/aaltonorms/figures/'


norm = "aaltoprod"

#Get features

#Get data from the big excel file
LUT = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', 
                    encoding='utf-8', 
                    header=0, index_col=0)

#Exclude homonyms, verbs and abstract words
#LUT = LUT[LUT['action_words'] ==0]
#LUT = LUT[LUT['category']!="abstract_mid"]
#LUT = LUT[LUT['category']!="abstract_high"]
# = LUT[LUT['category'].str.contains("abstract")]

norm_vocab = pd.read_csv(normpath + norm + '/' + 'vocab.csv', 
                          encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)


norm_vecs = pd.read_csv(normpath + norm + '/' + 'vectors.csv', 
                         encoding='utf-8', delimiter = '\t', 
                         header=None, index_col=None)

features = pd.read_csv('/m/nbe/project/aaltonorms/data/aaltoprod/features.csv', 
                       header=None)

picks = LUT[LUT[norm].notnull()]
picks = picks.sort_values(by=["category"])

abstract = picks[picks['category'].str.contains("abstract")]

#Set word label as index
norm_vecs.set_index(norm_vocab.index, inplace=True)
norm_vecs = norm_vecs.loc[picks[norm]]
abstract_vecs = norm_vecs.loc[abstract[norm]]

#Check dimensions
print("Number words: " + str(len(norm_vecs)))


def get_distances(norms):
    distmat = squareform(pdist(norms, metric=dist))
    distmat_full = list(distmat)
    tri = np.tril_indices(norms.shape[0]) #For denoting lower triangular
    distmat[tri] = np.nan
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
    ax = plt.imshow(cats, cmap='Paired', interpolation='nearest', 
                    extent=[0,5,0,1], aspect=100)
    ax = remove_ticks(ax)
    return ax
    

def plot_norms(A,label_A, cats=None):
    plt.figure(figsize=(10,3))    
    ax1 = plt.subplot(1,1,1)
    plt.title(label_A)
    ax1 = remove_ticks(ax1)
    plt.imshow(get_distances(A)[0], cmap="plasma", interpolation='nearest')
    plt.clim(0, 1);
    plt.colorbar()
    plt.savefig(figure_dir + label_A + "_dissimilarity.pdf", 
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
    #plt.show()
    index = leaves_list(Z)
    return index, Z


norm_index,norm_Z = hierarchical_clustering(abstract_vecs.values, 
                                              abstract['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norm + "_hierarchical_clustering.pdf", 
            format='pdf', dpi=1000, bbox_inches='tight')

plot_norms(norm_vecs.values, norm)

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
    n = [list(group) for key, group in groupby(D2['ivl'],
         lambda x: x in D1['ivl'])]
    return n

#Count number of features
picks['NOF'] = norm_vecs.astype(bool).sum(axis=1).values

#Identify distinctive features (occur in less than 2 concepts)
dF = norm_vecs.astype(bool).sum(axis=0)<3
#Count NodF for each concept
picks['NOdF'] = norm_vecs[dF.index[dF]].astype(bool).sum(axis=1).values

#Identify number of shared features (occur in less in 3 or more concepts)
sF = norm_vecs.astype(bool).sum(axis=0)>2
#Count NOsF for each concept
picks['NOsF']= norm_vecs[sF.index[sF]].astype(bool).sum(axis=1).values
