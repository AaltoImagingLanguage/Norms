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
#import argparse
#import os

normpath = '/m/nbe/project/aaltonorms/data/'
#Distance metric
dist = "cosine"
#Directory for the figures to be created
figure_dir = '/m/nbe/project/aaltonorms/figures/'
#norms = ["aaltoprod", "cslb", "mcrae", "vinson", "w2v_eng", "w2v_fin"]
norms = ["aaltoprod", "cslb", "vinson", "w2v_eng", "w2v_fin"]
bonferroni_factor = 10

#Get data from the big excel file
LUT = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', 
                    encoding='utf-8', 
                    header=0, index_col=0)

#Exclude homonyms, verbs and abstract words
LUT = LUT[LUT['action_words']==0]
LUT = LUT[LUT['category']!="abstract_mid"]
LUT = LUT[LUT['category']!="abstract_high"]

#Get vocab for all norm sets
n1_vocab = pd.read_csv(normpath + norms[0] + '/' + 'vocab.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=0)
n2_vocab = pd.read_csv(normpath + norms[1] + '/' + 'vocab.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=0)
n3_vocab = pd.read_csv(normpath + norms[2] + '/' + 'vocab.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=0)
n4_vocab = pd.read_csv(normpath + norms[3] + '/' + 'vocab.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=0)
n5_vocab = pd.read_csv(normpath + norms[4] + '/' + 'vocab.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=0)
#n6_vocab = pd.read_csv(normpath + norms[5] + '/' + 'vocab.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=0)

#Get vectors for all norm sets
n1_vecs = pd.read_csv(normpath + norms[0] + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=None)
n2_vecs = pd.read_csv(normpath + norms[1] + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=None)
n3_vecs = pd.read_csv(normpath + norms[2] + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=None)
n4_vecs = pd.read_csv(normpath + norms[3] + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=None)
n5_vecs = pd.read_csv(normpath + norms[4] + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=None)
#n6_vecs = pd.read_csv(normpath + norms[5] + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', header=None, index_col=None)

#Pick concepts that are available in all norms
#picks = LUT[LUT[norms[0]].notnull() & LUT[norms[1]].notnull() &
#            LUT[norms[2]].notnull() & LUT[norms[3]].notnull() & 
#            LUT[norms[4]].notnull() & LUT[norms[5]].notnull()]

picks = LUT[LUT[norms[0]].notnull() & LUT[norms[1]].notnull() &
            LUT[norms[2]].notnull() & LUT[norms[3]].notnull() & 
            LUT[norms[4]].notnull()]

picks = picks.sort_values(by=["category"])

#Set word label as index
n1_vecs.set_index(n1_vocab.index, inplace=True)
n2_vecs.set_index(n2_vocab.index, inplace=True)
n3_vecs.set_index(n3_vocab.index, inplace=True)
n4_vecs.set_index(n4_vocab.index, inplace=True)
n5_vecs.set_index(n5_vocab.index, inplace=True)
#n6_vecs.set_index(n6_vocab.index, inplace=True)

#Select norm vectors that can be found in picks
n1_vecs = n1_vecs.loc[picks[norms[0]]]
n2_vecs = n2_vecs.loc[picks[norms[1]]]
n3_vecs = n3_vecs.loc[picks[norms[2]]]
n4_vecs = n4_vecs.loc[picks[norms[3]]]
n5_vecs = n5_vecs.loc[picks[norms[4]]]
#n6_vecs = n6_vecs.loc[picks[norms[5]]]

#Check dimensions
#assert len(n1_vecs) == len(n2_vecs) == len(n3_vecs) == len(n4_vecs) == len(n5_vecs) == len(n6_vecs)
assert len(n1_vecs) == len(n2_vecs) == len(n3_vecs) == len(n4_vecs) == len(n5_vecs) 
print("Number of overlapping words: " + str(len(n1_vecs)))


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
    p_corr = pval *bonferroni_factor
    print(label_A + " vs. " + label_B + "  rho is: " + 
        str(round(rho,2)) + ", pvalue = " + str(round(pval,5)))
    if p_corr < 0.001:
       print("Corrected p < 0.001")
    plt.savefig(figure_dir + label_A + "_" + label_B + "_all2all_norm_comparison.pdf", 
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

n1_index, n1_Z = hierarchical_clustering(n1_vecs.values, picks['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norms[0] + "_all2all_hierarchical_clustering.pdf", format='pdf', dpi=1000, bbox_inches='tight')

n2_index,n2_Z = hierarchical_clustering(n2_vecs.values, picks['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norms[1] + "_all2all_hierarchical_clustering.pdf", format='pdf', dpi=1000, bbox_inches='tight')

n3_index, n3_Z = hierarchical_clustering(n3_vecs.values, picks['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norms[2] + "_all2all_hierarchical_clustering.pdf", format='pdf', dpi=1000, bbox_inches='tight')

n4_index,n4_Z = hierarchical_clustering(n4_vecs.values, picks['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norms[3] + "_all2all_hierarchical_clustering.pdf", format='pdf', dpi=1000, bbox_inches='tight')

n5_index, n5_Z = hierarchical_clustering(n5_vecs.values, picks['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norms[4] + "_all2all_hierarchical_clustering.pdf", format='pdf', dpi=1000, bbox_inches='tight')

#n6_index,n6_Z = hierarchical_clustering(n6_vecs.values, picks['eng_name'].tolist(),'8.')
#plt.savefig(figure_dir + norms[5] + "_all2all_hierarchical_clustering.pdf", format='pdf', dpi=1000, bbox_inches='tight')


compare_norms(n1_vecs.values, n2_vecs.values, norms[0], norms[1])
compare_norms(n1_vecs.values, n3_vecs.values, norms[0], norms[2])
compare_norms(n1_vecs.values, n4_vecs.values, norms[0], norms[3])
compare_norms(n1_vecs.values, n5_vecs.values, norms[0], norms[4])
#compare_norms(n1_vecs.values, n6_vecs.values, norms[0], norms[5])
compare_norms(n2_vecs.values, n3_vecs.values, norms[1], norms[2])
compare_norms(n2_vecs.values, n4_vecs.values, norms[1], norms[3])
compare_norms(n2_vecs.values, n5_vecs.values, norms[1], norms[4])
#compare_norms(n2_vecs.values, n6_vecs.values, norms[1], norms[5])
compare_norms(n3_vecs.values, n4_vecs.values, norms[2], norms[3])
compare_norms(n3_vecs.values, n5_vecs.values, norms[2], norms[4])
#compare_norms(n3_vecs.values, n6_vecs.values, norms[2], norms[5])
compare_norms(n4_vecs.values, n5_vecs.values, norms[3], norms[4])
#compare_norms(n4_vecs.values, n6_vecs.values, norms[3], norms[5])
#compare_norms(n5_vecs.values, n6_vecs.values, norms[4], norms[5])


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
