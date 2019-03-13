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
from scipy.stats import spearmanr
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import dendrogram, leaves_list
import argparse
import os

normpath = '/m/nbe/project/aaltonorms/data/'
#Distance metric
dist = "cosine"
#Directory for the figures to be created
figure_dir = '/m/nbe/project/aaltonorms/figures/'

parser = argparse.ArgumentParser(description='Learn a mapping from one norm dataset to another')
parser.add_argument('norm1', type=str, help='Norm data set 1')
parser.add_argument('norm2', type=str, help='Norm data set 2')
parser.add_argument('--reg', action='store_true', help='Whether to use regularization')
args = parser.parse_args()

print('Comparing', os.path.basename(args.norm1), 'to', os.path.basename(args.norm2))
norm1 = args.norm1
norm2 = args.norm2

#Get data from the big excel file
LUT = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', encoding='utf-8', 
                       header=0, index_col=0)

#Exclude homonyms
LUT = LUT[LUT['homonym_all']==0]

norm1_vocab = pd.read_csv(normpath + norm1 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norm2_vocab = pd.read_csv(normpath + norm2 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norm1_vecs = pd.read_csv(normpath + norm1 + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', 
                         header=None, index_col=None)

norm2_vecs = pd.read_csv(normpath + norm2 + '/' + 'vectors.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=None)

picks = LUT[LUT[norm1].notnull() & LUT[norm2].notnull()]
picks = picks.sort_values(by=["category"])

#Set word label as index
norm1_vecs.set_index(norm1_vocab.index, inplace=True)
norm2_vecs.set_index(norm2_vocab.index, inplace=True)

norm1_vecs = norm1_vecs.loc[picks[norm1]]
norm2_vecs = norm2_vecs.loc[picks[norm2]]


#Check dimensions
assert len(norm1_vecs) == len(norm2_vecs)
print("Number of overlapping words: " + str(len(norm1_vecs)))


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
    ax = plt.imshow(cats, cmap='Paired', interpolation='nearest', extent=[0,5,0,1], aspect=100)
    ax = remove_ticks(ax)
    return ax
    

def compare_norms(A,B, label_A, label_B,  cats=None):
    plt.figure(figsize=(10,3))    
    ax1 = plt.subplot(1,2,1)
    plt.title(label_A)
    ax1 = remove_ticks(ax1)
    plt.imshow(get_distances(A)[0], cmap="plasma", interpolation='nearest')
    ax2 = plt.subplot(1,2,2)
    ax2 = remove_ticks(ax2)
    plt.title(label_B)
    plt.imshow(get_distances(B)[0], cmap="plasma", interpolation='nearest')
    plt.clim(0, 1);
    plt.colorbar(ax=[ax1, ax2])
    rho, pval = spearmanr(get_distances(A)[1], get_distances(B)[1])
   
    print(label_A + " vs. " + label_B + "  rho is: " + 
        str(round(rho,2)) + ", pvalue = " + str(round(pval,5)))
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
plt.savefig(figure_dir + norm1 + "_hierarchical_clustering.pdf", 
            format='pdf', dpi=1000, bbox_inches='tight')
norm2_index,norm2_Z = hierarchical_clustering(norm2_vecs.values, picks['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norm2 + "_hierarchical_clustering.pdf", 
            format='pdf', dpi=1000, bbox_inches='tight')

compare_norms(norm1_vecs.values, norm2_vecs.values, norm1,norm2)

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
