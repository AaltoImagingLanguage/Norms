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

dist = "cosine"
norms1 = "w2v_eng"
norms2 = "vinson"
normpath = '/m/nbe/project/aaltonorms/data/'

#Directory for the figures to be created
figure_dir = '/m/nbe/project/aaltonorms/figures/'


#Get data from files
LUT = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', encoding='utf-8', 
                       header=0, index_col=0)
norms1_vocab = pd.read_csv(normpath + norms1 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norms2_vocab = pd.read_csv(normpath + norms2 + '/' + 'vocab.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=0)

norms1_vecs = pd.read_csv(normpath + norms1 + '/' + 'vectors.csv', encoding='utf-8', delimiter = '\t', 
                         header=None, index_col=None)

norms2_vecs = pd.read_csv(normpath + norms2 + '/' + 'vectors.csv', encoding='utf-8', 
                         delimiter = '\t', header=None, index_col=None)


print("Norms compared: " + norms1 + " " + norms2)

picks = LUT[LUT[norms1].notnull() & LUT[norms2].notnull()]

picks = picks.sort_values(by=["category"])

#Set word label as index
norms1_vecs.set_index(norms1_vocab.index, inplace=True)
norms2_vecs.set_index(norms2_vocab.index, inplace=True)

norms1_vecs = norms1_vecs.loc[picks[norms1]]
norms2_vecs = norms2_vecs.loc[picks[norms2]]


#Check dimensions
assert len(norms1_vecs) == len(norms2_vecs)
print("Number of overlapping words: " + str(len(norms1_vecs)))


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
        str(round(rho,2)) + ", pvalue = " + str(round(pval,2)))
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


norms1_index,norms1_Z = hierarchical_clustering(norms1_vecs.values, picks['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norms1 + "_hierarchical_clustering.pdf", 
            format='pdf', dpi=1000, bbox_inches='tight')
norms2_index,norms2_Z = hierarchical_clustering(norms2_vecs.values, picks['eng_name'].tolist(),'8.')
plt.savefig(figure_dir + norms2 + "_hierarchical_clustering.pdf", 
            format='pdf', dpi=1000, bbox_inches='tight')

compare_norms(norms1_vecs.values, norms2_vecs.values, norms1,norms2)





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
#
#n = get_different_clusters(Z_cslb)
#m = get_different_clusters(Z_vinson)
#
#clusters_cslb = []
#for k in n:
#    group = []
#    for i in k:
#        group.append(int(i))
#    clusters_cslb.append(group)
##print(clusters_cslb)
#
#clusters_vinson = []
#for k in m:
#    group = []
#    for i in k:
#        group.append(int(i))
#    clusters_vinson.append(group)
##print(clusters_vinson)
#
#cslb_hierarchial_word = []
#cslb_hierarchial_group = []
#for k in clusters_cslb:
#    group =[]
#    for i in k:
#        group.append(cslb_list[i])
#        #cslb_hierarchial_word.append(cslb_list[i])
#    cslb_hierarchial_group.append(group)
##print(cslb_hierarchial_group)
#
#vinson_hierarchial_word = []
#vinson_hierarchial_group = []
#for k in clusters_vinson:
#    group = []
#    for i in k:
#        group.append(cslb_list[i])
#        #vinson_hierarchial_word.append(cslb_list[i])
#    vinson_hierarchial_group.append(group)
##print(vinson_hierarchial_group)
#
#cslb_hierarchial_i = []
#a = 0
#while a < 8:
##    if 11 < a and a < 20:
##        a += 1
##    else:
#    for name in cslb_hierarchial_group[a]:
#        cslb_hierarchial_word.append(name)
#        i=0
#        for row in cslb_vocab.loc[:,"cslb"]:
#            if (name == cslb_vocab.loc[i,"cslb"]):
#                cslb_hierarchial_i.append(i)
#                break
#            else:
#                i += 1
#    a += 1            
##print(cslb_hierarchial_i) 
#cslb_hierarchial = cslb.iloc[cslb_hierarchial_i]
##print(cslb_hierarchial)
#
#vinson_hierarchial_i = []
#b = 0
#while b < 3:
#    for name in vinson_hierarchial_group[b]:
#        vinson_hierarchial_word.append(name)
#        i=0
#        for row in vinson_vocab.loc[:,"cslb"]:
#            if (name == vinson_vocab.loc[i,"cslb"]):
#                vinson_hierarchial_i.append(i)
#                break
#            else:
#                i += 1
#    b += 1
##print(vinson_hierarchial_i) 
#vinson_hierarchial = vinson.iloc[vinson_hierarchial_i]
##print(vinson_hierarchial)

#compare_norms(cslb_hierarchial,vinson_hierarchial,'cslb','vinson','eläimet')
#
#hierarchical_clustering(cslb_hierarchial,cslb_hierarchial_word,'14.')
#hierarchical_clustering(vinson_hierarchial,vinson_hierarchial_word,'14.')

