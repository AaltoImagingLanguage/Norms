#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:05:14 2018

@author: juurakj1
"""
#from scipy.io import loadmat
import pandas as pd
import pandas
from scipy.spatial.distance import squareform, pdist
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr
import scipy.cluster.hierarchy as shc
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from scipy import cluster
from scipy.cluster import hierarchy


normpath = '/m/nbe/project/aaltonorms/data/'
#Directory for the figures to be created
figure_dir = '/m/home/home1/18/juurakj1/unix/aaltonorms/figures/'

#These are the files for different norm sets
#filenames = ['cslb_aaltoOverlap', 'vinson_aaltoOverlap', 'aalto85_aaltoOverlap',
#'ginter_aaltoOverlap', 'cmu_aaltoOverlap', 'aalto300', 'aalto300_cslbOverlap',
#'aalto300_vinsonOverlap','aalto300_aalto85Overlap', 'aalto300_cmuOverlap', 
#'vinson_ginterOverlap', 'cslb_ginterOverlap', 'ginter_vinsonOverlap', 'ginter_cslbOverlap']

def get_normdata(filename):   
    df = pandas.read_table(normpath + filename + '.csv', encoding='utf-8', 
    header=None, index_col=0)
    return df


cslb = pandas.read_table(normpath + 'cslb/' + 'vectors.csv', encoding='utf-8', 
                         header=None, index_col=None)

vinson = pandas.read_table(normpath + 'vinson/' + 'vectors.csv', encoding='utf-8', 
                         header=None, index_col=None)

cmu = pandas.read_table(normpath + 'cmu/' + 'vectors.csv', encoding='utf-8', 
                         header=None, index_col=None)

aaltoprod = pandas.read_table(normpath + 'aaltoprod/' + 'vectors.csv', encoding='utf-8', 
                         header=None, index_col=None)

aalto85 = pandas.read_table(normpath + 'aalto85/' + 'vectors.csv', encoding='utf-8', 
                         header=None, index_col=None)

mcrae = pandas.read_table(normpath + 'mcrae/' + 'vectors.csv', encoding='utf-8', 
                         header=None, index_col=None, sep = ';')

w2v_fin = pandas.read_table(normpath + 'w2v_fin/' + 'vectors.csv', encoding='utf-8', 
                         header=None, index_col=None, sep = '\t')

w2v_eng = pandas.read_table(normpath + 'w2v_eng/' + 'vectors.csv', encoding='utf-8', 
                         header=None, index_col=None, sep = '\t')

cslb_vocab = pandas.read_table(normpath + 'cslb/' + 'vocab.csv', encoding='utf-8', 
                         header=None, index_col=None)

vinson_vocab = pandas.read_table(normpath + 'vinson/' + 'vocab.csv', encoding='utf-8', 
                         header=None, index_col=None)

cmu_vocab = pandas.read_table(normpath + 'cmu/' + 'vocab.csv', encoding='utf-8', 
                         header=None, index_col=None)

aaltoprod_vocab = pandas.read_table(normpath + 'aaltoprod/' + 'vocab.csv', encoding='utf-8', 
                         header=None, index_col=None)

aalto85_vocab = pandas.read_table(normpath + 'aalto85/' + 'vocab.csv', encoding='utf-8', 
                         header=None, index_col=None)

mcrae_vocab = pandas.read_table(normpath + 'mcrae/' + 'vocab.csv', encoding='utf-8', 
                         header=None, index_col=None)

w2v_fin_vocab = pandas.read_table(normpath + 'w2v_fin/' + 'vocab.csv', encoding='utf-8', 
                         header=None, index_col=None)

w2v_eng_vocab = pandas.read_table(normpath + 'w2v_eng/' + 'vocab.csv', encoding='utf-8', 
                         header=None, index_col=None)

cslb_corr = cslb_vocab = pandas.read_table(normpath + 'cslb/' + 'correspondence.csv', encoding='utf-8', 
                         header=0, index_col=None)

vinson_corr = vinson_vocab = pandas.read_table(normpath + 'vinson/' + 'correspondence.csv', encoding='utf-8', 
                         header=0, index_col=None)

cmu_corr = cmu_vocab = pandas.read_table(normpath + 'cmu/' + 'correspondence.csv', encoding='utf-8', 
                         header=0, index_col=None)

aaltoprod_corr = aaltoprod_vocab = pandas.read_table(normpath + 'aaltoprod/' + 'correspondence.csv', encoding='utf-8', 
                         header=0, index_col=None)

aalto85_corr = aalto85_vocab = pandas.read_table(normpath + 'aalto85/' + 'correspondence.csv', encoding='utf-8', 
                         header=0, index_col=None)

mcrae_corr = mcrae_vocab = pandas.read_table(normpath + 'mcrae/' + 'correspondence.csv', encoding='utf-8', 
                         header=0, index_col=None)

w2v_fin_corr = w2v_fin_vocab = pandas.read_table(normpath + 'w2v_fin/' + 'correspondence.csv', encoding='utf-8', 
                         header=0, index_col=None)

w2v_eng_corr = w2v_eng_vocab = pandas.read_table(normpath + 'w2v_eng/' + 'correspondence.csv', encoding='utf-8', 
                         header=0, index_col=None)


def get_common_vectors(A,B,A_vocab,B_vocab,A_corr,B_corr,A_name,B_name):
    A_list = []
    B_list = []
    
    for name in A_corr[B_name]:      
        if isinstance(name, str):
            B_list.append(name)
                    
    for name in B_corr[A_name]:
        if isinstance(name, str):
            A_list.append(name)
     
    #A_list = A_list[::2]    
    #B_list = B_list[::2]    
    A_index = []
    B_index = []
    
    for name in A_list:
        i = 0
        for row in A_vocab.loc[:, A_name]:
            if (name == A_vocab.loc[i, A_name]):
                A_index.append(i)
                break
            else:
                i+=1
    A_vectors = A.loc[A_index]

    for name in B_list:
        i = 0
        for row in B_vocab.loc[:, B_name]:
            if (name == B_vocab.loc[i, B_name]):
                B_index.append(i)
                break
            else:
                i+=1
    B_vectors = B.loc[B_index]
    return A_vectors,B_vectors,A_list,B_list

def get_distances(norms):
    distmat = squareform(pdist(norms, metric="cosine"))
    distmat_full = list(distmat)
    tri = np.tril_indices(norms.shape[0]) #For denoting lower triangular
    distmat[tri] = np.nan
    distvector = np.asarray(distmat.reshape(-1)) #Take upper triangular                                                     #and reshape
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
    

def compare_norms(A,B, label_A, label_B, label, cats=None):
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
   
    print('Pvalue:')
    print(pval)
    print(label_A + " vs. " + label_B + "  rho is: " + 
        str(round(rho,2)) + ", pvalue = " + str(round(pval,2)))
    plt.savefig(figure_dir + label_A + label_B + label + "_production_norm_comparison.pdf", 
                format='pdf', dpi=1000, bbox_inches='tight')


index = []
def hierarchical_clustering(norms,labels,font):
    Z = hac.linkage(norms, method='complete', metric='cosine')   
    labels = labels
    # Plot dendogram
    plt.figure(figsize=(50, 5))
    plt.title('Hierarchical Clustering Dendrogram', fontsize=40)
    plt.xlabel('Norm', fontsize=40)
    plt.ylabel('Distance', fontsize=40)
    hac.dendrogram(
        Z,
        labels=labels,
        #p =5,
        #truncate_mode='level',
        leaf_rotation = 90.,  # rotates the x axis labels
        leaf_font_size= font,  # font size for the x axis labels
    )
    plt.show()
    #plt.savefig(figure_dir + label_A + "+(" + label_B + ")_" + label + ".png", 
     #           format='png', bbox_inches='tight')
    index = leaves_list(Z)
    return index,Z

def reorder_vectors(A,B,A_index,B_index,A_list,B_list,A_vocab,B_vocab,A_name,B_name):
    
    A_hierarchial_word = []
    B_hierarchial_word = []
    for i in A_index:
        A_hierarchial_word.append(A_list[i])
    for i in B_index:
        B_hierarchial_word.append(B_list[i])
    
    A_hierarchial_word = A_hierarchial_word[:100]
    #print(A_hierarchial_word)
    B_hierarchial_word = B_hierarchial_word[:100]
    
    A_hierarchial_i = []
    for name in A_hierarchial_word:
        i=0
        for row in A_vocab.loc[:,A_name]:
            if (name == A_vocab.loc[i,A_name]):
                A_hierarchial_i.append(i)
                break
            else:
                i += 1
    A_hierarchial_i = A_hierarchial_i[:100]
    #print(A_hierarchial_i)
    A_hierarchial = A.iloc[A_hierarchial_i]
    
    B_hierarchial_i = []
    for name in B_hierarchial_word:
        i=0
        for row in B_vocab.loc[:,B_name]:
            if (name == B_vocab.loc[i,B_name]):
                B_hierarchial_i.append(i)
                break
            else:
                i += 1
    B_hierarchial_i = B_hierarchial_i[:100]
    B_hierarchial = B.iloc[B_hierarchial_i]
    return A_hierarchial,B_hierarchial,B_hierarchial_word

def get_category(A,B,A_vocab,B_vocab,A_corr,B_corr,A_name,B_name,category):
    A_list = []
    B_list = []
    
    a = 0
    for a in range(len(A_vocab)):
        if (category == A_corr.loc[a,'category']):
            name = A_corr.loc[a,B_name]      
            if isinstance(name, str):
                B_list.append(name)
                a += 1
        else:
            a+=1
            
    b = 0
    for b in range(len(B_vocab)):
        if (category == B_corr.loc[b,'category']):
            name = B_corr.loc[b,A_name]      
            if isinstance(name, str):
                A_list.append(name)
                b += 1
        else:
            b+=1
        
    #A_list = A_list[::2]    
    #B_list = B_list[::2]    
    A_index = []
    B_index = []
    
    for name in A_list:
        i = 0
        for row in A_vocab.loc[:, A_name]:
            if (name == A_vocab.loc[i, A_name]):
                A_index.append(i)
                break
            else:
                i+=1
    A_vectors = A.loc[A_index]

    for name in B_list:
        i = 0
        for row in B_vocab.loc[:, B_name]:
            if (name == B_vocab.loc[i, B_name]):
                B_index.append(i)
                break
            else:
                i+=1
    B_vectors = B.loc[B_index]
    return A_vectors,B_vectors,A_list,B_list




'''
#CSLB vs Vinson 
cslb_vectors,vinson_vectors,cslb_list,vinson_list=get_common_vectors(
        cslb,vinson,cslb_vocab,vinson_vocab,cslb_corr,vinson_corr,'cslb','vinson')


cslb_index,Z_cslb = hierarchical_clustering(cslb_vectors,cslb_list,'20.')
vinson_index,Z_vinson = hierarchical_clustering(vinson_vectors,cslb_list,'20.')

cslb_hierarchial,vinson_hierarchial = reorder_vectors(cslb,vinson,
                                                      cslb_index,vinson_index,
                                                      cslb_list,vinson_list,
                                                      cslb_vocab,vinson_vocab,
                                                      'cslb','vinson')

#Plot comparison in hierarchical order
compare_norms(cslb_hierarchial,vinson_hierarchial,'CSLB','Vinson','common')
'''


'''
#Tutkitaan tarkemmin klustereita
cslb_vectors,vinson_vectors,cslb_list,vinson_list=get_category(
        cslb,vinson,cslb_vocab,vinson_vocab,cslb_corr,vinson_corr,'cslb','vinson','vehicle')

cslb_index,Z_cslb = hierarchical_clustering(cslb_vectors,cslb_list,'50.')
vinson_index,Z_vinson = hierarchical_clustering(vinson_vectors,cslb_list,'50.')

cslb_hierarchial,vinson_hierarchial = reorder_vectors(cslb,vinson,
                                                      cslb_index,vinson_index,
                                                      cslb_list,vinson_list,
                                                      cslb_vocab,vinson_vocab,
                                                      'cslb','vinson')

#Plot comparison in hierarchical order
compare_norms(cslb_hierarchial,vinson_hierarchial,'CSLB','Vinson','vehicles')
'''

'''
#korpukset
w2v_fin_vectors,w2v_eng_vectors,w2v_fin_list,w2v_eng_list=get_common_vectors(
        w2v_fin,w2v_eng,w2v_fin_vocab,w2v_eng_vocab,w2v_fin_corr,w2v_eng_corr,'w2v_fin','w2v_eng')

w2v_fin_index,Z_w2v_fin = hierarchical_clustering(w2v_fin_vectors,w2v_eng_list,'20.')
w2v_eng_index,Z_w2v_eng = hierarchical_clustering(w2v_eng_vectors,w2v_eng_list,'20.')

#w2v_fin_index = w2v_fin_index[:100]
#w2v_eng_index = w2v_eng_index[:100]

w2v_fin_hierarchial,w2v_eng_hierarchial,w2v_eng_list = reorder_vectors(w2v_fin,w2v_eng,
                                                      w2v_fin_index,w2v_eng_index,
                                                      w2v_fin_list,w2v_eng_list,
                                                      w2v_fin_vocab,w2v_eng_vocab,
                                                      'w2v_fin','w2v_eng')

w2v_fin_index,Z_w2v_fin = hierarchical_clustering(w2v_fin_vectors,w2v_eng_list,'20.')
w2v_eng_index,Z_w2v_eng = hierarchical_clustering(w2v_eng_vectors,w2v_eng_list,'20.')

#Plot comparison in hierarchical order
compare_norms(w2v_fin_hierarchial,w2v_eng_hierarchial,'Korpus_fin','Korpus_eng','common')
'''


'''
#Tutkitaan tarkemmin klustereita
w2v_fin_vectors,w2v_eng_vectors,w2v_fin_list,w2v_eng_list=get_category(
        w2v_fin,w2v_eng,w2v_fin_vocab,w2v_eng_vocab,w2v_fin_corr,w2v_eng_corr,'w2v_fin','w2v_eng','vehicle')

w2v_fin_index,Z_w2v_fin = hierarchical_clustering(w2v_fin_vectors,w2v_eng_list,'50.')
w2v_eng_index,Z_w2v_eng = hierarchical_clustering(w2v_eng_vectors,w2v_eng_list,'50.')

w2v_fin_hierarchial,w2v_eng_hierarchial = reorder_vectors(w2v_fin,w2v_eng,
                                                      w2v_fin_index,w2v_eng_index,
                                                      w2v_fin_list,w2v_eng_list,
                                                      w2v_fin_vocab,w2v_eng_vocab,
                                                      'w2v_fin','w2v_eng')

#Plot comparison in hierarchical order
compare_norms(w2v_fin_hierarchial,w2v_eng_hierarchial,'Korpus_fin','Korpus_eng','vehicles')
'''


'''
#CSLB vs w2v_eng
cslb_vectors,w2v_eng_vectors,cslb_list,w2v_eng_list=get_common_vectors(
        cslb,w2v_eng,cslb_vocab,w2v_eng_vocab,cslb_corr,w2v_eng_corr,'cslb','w2v_eng')

cslb_index,Z_cslb = hierarchical_clustering(cslb_vectors,cslb_list,'8.')
w2v_eng_index,Z_w2v_eng = hierarchical_clustering(w2v_eng_vectors,cslb_list,'8.')

cslb_hierarchial,w2v_eng_hierarchial = reorder_vectors(cslb,w2v_eng,
                                                      cslb_index,w2v_eng_index,
                                                      cslb_list,w2v_eng_list,
                                                      cslb_vocab,w2v_eng_vocab,
                                                      'cslb','w2v_eng')

#Plot comparison in hierarchical order
compare_norms(cslb_hierarchial,w2v_eng_hierarchial,'CSLB','W2V_eng','complete')
'''

'''
#Tutkitaan tarkemmin klustereita
cslb_vectors,w2v_eng_vectors,cslb_list,w2v_eng_list=get_category(
        cslb,w2v_eng,cslb_vocab,w2v_eng_vocab,cslb_corr,w2v_eng_corr,'cslb','w2v_eng','clothing')

cslb_index,Z_cslb = hierarchical_clustering(cslb_vectors,cslb_list,'40.')
w2v_eng_index,Z_w2v_eng = hierarchical_clustering(w2v_eng_vectors,cslb_list,'40.')

cslb_hierarchial,w2v_eng_hierarchial = reorder_vectors(cslb,w2v_eng,
                                                      cslb_index,w2v_eng_index,
                                                      cslb_list,w2v_eng_list,
                                                      cslb_vocab,w2v_eng_vocab,
                                                      'cslb','w2v_eng')

#Plot comparison in hierarchical order
compare_norms(cslb_hierarchial,w2v_eng_hierarchial,'CSLB','W2V_eng','clothing')
'''

'''
#Vinson vs w2v_eng
vinson_vectors,w2v_eng_vectors,vinson_list,w2v_eng_list=get_common_vectors(
        vinson,w2v_eng,vinson_vocab,w2v_eng_vocab,vinson_corr,w2v_eng_corr,'vinson','w2v_eng')

vinson_index,Z_vinson = hierarchical_clustering(vinson_vectors,w2v_eng_list,'20.')
w2v_eng_index,Z_w2v_eng = hierarchical_clustering(w2v_eng_vectors,w2v_eng_list,'20.')

vinson_hierarchial,w2v_eng_hierarchial = reorder_vectors(vinson,w2v_eng,
                                                      vinson_index,w2v_eng_index,
                                                      vinson_list,w2v_eng_list,
                                                      vinson_vocab,w2v_eng_vocab,
                                                      'vinson','w2v_eng')

#Plot comparison in hierarchical order
compare_norms(vinson_hierarchial,w2v_eng_hierarchial,'Vinson','Korpus_eng','complete')
'''

'''
#Tutkitaan tarkemmin klustereita
vinson_vectors,w2v_eng_vectors,vinson_list,w2v_eng_list=get_category(
        vinson,w2v_eng,vinson_vocab,w2v_eng_vocab,vinson_corr,w2v_eng_corr,'vinson','w2v_eng','vehicle')

vinson_index,Z_vinson = hierarchical_clustering(vinson_vectors,w2v_eng_list,'50.')
w2v_eng_index,Z_w2v_eng = hierarchical_clustering(w2v_eng_vectors,w2v_eng_list,'50.')

vinson_hierarchial,w2v_eng_hierarchial = reorder_vectors(vinson,w2v_eng,
                                                      vinson_index,w2v_eng_index,
                                                      vinson_list,w2v_eng_list,
                                                      vinson_vocab,w2v_eng_vocab,
                                                      'vinson','w2v_eng')

#Plot comparison in hierarchical order
compare_norms(vinson_hierarchial,w2v_eng_hierarchial,'Vinson','Korpus_eng','vehicles')
'''

'''
#Aaltoprod vs Aalto85
aaltoprod_vectors,aalto85_vectors,aaltoprod_list,aalto85_list=get_common_vectors(
        aaltoprod,aalto85,aaltoprod_vocab,aalto85_vocab,aaltoprod_corr,aalto85_corr,'aaltoprod','aalto85')

aaltoprod_index,Z_aaltoprod = hierarchical_clustering(aaltoprod_vectors,aaltoprod_list,'20.')
aalto85_index,Z_aalto85 = hierarchical_clustering(aalto85_vectors,aaltoprod_list,'20.')

aaltoprod_hierarchial,aalto85_hierarchial = reorder_vectors(aaltoprod,aalto85,
                                                      aaltoprod_index,aalto85_index,
                                                      aaltoprod_list,aalto85_list,
                                                      aaltoprod_vocab,aalto85_vocab,
                                                      'aaltoprod','aalto85')

#Plot comparison in hierarchical order
compare_norms(aaltoprod_hierarchial,aalto85_hierarchial,'Aaltoprod','Aalto85','complete')
'''

'''
#Tutkitaan tarkemmin klustereita
aaltoprod_vectors,aalto85_vectors,aaltoprod_list,aalto85_list=get_category(
        aaltoprod,aalto85,aaltoprod_vocab,aalto85_vocab,aaltoprod_corr,aalto85_corr,'aaltoprod','aalto85','vehicle')

aaltoprod_index,Z_aaltoprod = hierarchical_clustering(aaltoprod_vectors,aaltoprod_list,'50.')
aalto85_index,Z_aalto85 = hierarchical_clustering(aalto85_vectors,aaltoprod_list,'50.')

aaltoprod_hierarchial,aalto85_hierarchial = reorder_vectors(aaltoprod,aalto85,
                                                      aaltoprod_index,aalto85_index,
                                                      aaltoprod_list,aalto85_list,
                                                      aaltoprod_vocab,aalto85_vocab,
                                                      'aaltoprod','aalto85')


#Plot comparison in hierarchical order
compare_norms(aaltoprod_hierarchial,aalto85_hierarchial,'Aaltoprod','Aalto85','vehicles')
'''

'''
#w2v_fin vs Aalto85
w2v_fin_vectors,aalto85_vectors,w2v_fin_list,aalto85_list=get_common_vectors(
        w2v_fin,aalto85,w2v_fin_vocab,aalto85_vocab,w2v_fin_corr,aalto85_corr,'w2v_fin','aalto85')

w2v_fin_index,Z_w2v_fin = hierarchical_clustering(w2v_fin_vectors,aalto85_list,'20.')
aalto85_index,Z_aalto85 = hierarchical_clustering(aalto85_vectors,aalto85_list,'20.')

w2v_fin_hierarchial,aalto85_hierarchial = reorder_vectors(w2v_fin,aalto85,
                                                      w2v_fin_index,aalto85_index,
                                                      w2v_fin_list,aalto85_list,
                                                      w2v_fin_vocab,aalto85_vocab,
                                                      'w2v_fin','aalto85')

#Plot comparison in hierarchical order
compare_norms(w2v_fin_hierarchial,aalto85_hierarchial,'Korpus_fin','Aalto85','complete')
'''

'''
#Tutkitaan tarkemmin klustereita
w2v_fin_vectors,aalto85_vectors,w2v_fin_list,aalto85_list=get_category(
        w2v_fin,aalto85,w2v_fin_vocab,aalto85_vocab,w2v_fin_corr,aalto85_corr,'w2v_fin','aalto85','tool')

w2v_fin_index,Z_w2v_fin = hierarchical_clustering(w2v_fin_vectors,aalto85_list,'50.')
aalto85_index,Z_aalto85 = hierarchical_clustering(aalto85_vectors,aalto85_list,'50.')

w2v_fin_hierarchial,aalto85_hierarchial = reorder_vectors(w2v_fin,aalto85,
                                                      w2v_fin_index,aalto85_index,
                                                      w2v_fin_list,aalto85_list,
                                                      w2v_fin_vocab,aalto85_vocab,
                                                      'w2v_fin','aalto85')

#Plot comparison in hierarchical order
compare_norms(w2v_fin_hierarchial,aalto85_hierarchial,'Korpus_fin','Aalto85','tools')
'''

'''
#Vinson vs McRae
vinson_vectors,mcrae_vectors,vinson_list,mcrae_list=get_common_vectors(
        vinson,mcrae,vinson_vocab,mcrae_vocab,vinson_corr,mcrae_corr,'vinson','mcrae')

vinson_index,Z_vinson = hierarchical_clustering(vinson_vectors,mcrae_list,'20.')
mcrae_index,Z_mcrae = hierarchical_clustering(mcrae_vectors,mcrae_list,'20.')

vinson_hierarchial,mcrae_hierarchial = reorder_vectors(vinson,mcrae,
                                                      vinson_index,mcrae_index,
                                                      vinson_list,mcrae_list,
                                                      vinson_vocab,mcrae_vocab,
                                                      'vinson','mcrae')

#Plot comparison in hierarchical order
compare_norms(vinson_hierarchial,mcrae_hierarchial,'Vinson','McRae','complete')
'''

'''
#Tutkitaan tarkemmin klustereita
vinson_vectors,mcrae_vectors,vinson_list,mcrae_list=get_category(
        vinson,mcrae,vinson_vocab,mcrae_vocab,vinson_corr,mcrae_corr,'vinson','mcrae','vehicle')

vinson_index,Z_vinson = hierarchical_clustering(vinson_vectors,mcrae_list,'50.')
mcrae_index,Z_mcrae = hierarchical_clustering(mcrae_vectors,mcrae_list,'50.')

vinson_hierarchial,mcrae_hierarchial = reorder_vectors(vinson,mcrae,
                                                      vinson_index,mcrae_index,
                                                      vinson_list,mcrae_list,
                                                      vinson_vocab,mcrae_vocab,
                                                      'vinson','mcrae')

#Plot comparison in hierarchical order
compare_norms(vinson_hierarchial,mcrae_hierarchial,'Vinson','McRae','vehicles')
'''


'''
#CSLB vs McRae
cslb_vectors,mcrae_vectors,cslb_list,mcrae_list=get_common_vectors(
        cslb,mcrae,cslb_vocab,mcrae_vocab,cslb_corr,mcrae_corr,'cslb','mcrae')

cslb_index,Z_cslb = hierarchical_clustering(cslb_vectors,cslb_list,'8.')
mcrae_index,Z_mcrae = hierarchical_clustering(mcrae_vectors,cslb_list,'8.')

cslb_hierarchial,mcrae_hierarchial = reorder_vectors(cslb,mcrae,
                                                      cslb_index,mcrae_index,
                                                      cslb_list,mcrae_list,
                                                      cslb_vocab,mcrae_vocab,
                                                      'cslb','mcrae')

#Plot comparison in hierarchical order
compare_norms(cslb_hierarchial,mcrae_hierarchial,'CSLB','McRae','common')
'''

'''
#CSLB vs CMU
cslb_vectors,cmu_vectors,cslb_list,cmu_list=get_common_vectors(
        cslb,cmu,cslb_vocab,cmu_vocab,cslb_corr,cmu_corr,'cslb','cmu')

cslb_index,Z_cslb = hierarchical_clustering(cslb_vectors,cslb_list,'8.')
cmu_index,Z_cmu = hierarchical_clustering(cmu_vectors,cslb_list,'8.')

cslb_hierarchial,cmu_hierarchial = reorder_vectors(cslb,cmu,
                                                      cslb_index,cmu_index,
                                                      cslb_list,cmu_list,
                                                      cslb_vocab,cmu_vocab,
                                                      'cslb','cmu')

#Plot comparison in hierarchical order
compare_norms(cslb_hierarchial,cmu_hierarchial,'CSLB','CMU','common')

#Tutkitaan tarkemmin klustereita... ei järkeä, koska cmu ei järjestyksessä
cslb_index,Z_cslb = hierarchical_clustering(cslb_vectors,cslb_list,'8.')
cmu_index,Z_cmu = hierarchical_clustering(cmu_vectors,cmu_list,'8.')    

cslb_hierarchial,cmu_hierarchial,cslb_hierarchial_word,cmu_hierarchial_word,x,y = get_different_clusters(
        Z_cslb,Z_cmu,cslb_list,cslb_vocab,cmu_vocab,'cslb',cslb,cmu)

compare_norms(cslb_hierarchial,cmu_hierarchial,'CSLB','Vinson','eläimet')

hierarchical_clustering(cslb_hierarchial,cslb_hierarchial_word,'14.')
hierarchical_clustering(cmu_hierarchial,cmu_hierarchial_word,'14.')
'''




'''
# A_list on se mitä käytetään molemmissa tekstinä
def get_different_clusters(Z_A,Z_B,A_list,A_vocab,B_vocab,A_name,A,B):
    D1=dendrogram(Z_A,
                  color_threshold=1,
                  p=40, # muokkaa
                  truncate_mode='lastp',
                  distance_sort='ascending')
    plt.close()
    D2=dendrogram(Z_A,
                  p=103, # muokkaa
                  truncate_mode='lastp',
                  distance_sort='ascending')
    plt.close()
    from itertools import groupby
    A_n = [list(group) for key, group in groupby(D2['ivl'],lambda x: x in D1['ivl'])]
    
    D1=dendrogram(Z_B,
                  color_threshold=1,
                  p=40, # muokkaa
                  truncate_mode='lastp',
                  distance_sort='ascending')
    plt.close()
    D2=dendrogram(Z_B,
                  p=103, # muokkaa
                  truncate_mode='lastp',
                  distance_sort='ascending')
    plt.close()
    from itertools import groupby
    B_n = [list(group) for key, group in groupby(D2['ivl'],lambda x: x in D1['ivl'])]

    clusters_A = []
    for k in A_n:
        group = []
        for i in k:
            group.append(int(i))
        clusters_A.append(group)
    
    clusters_B = []
    for k in B_n:
        group = []
        for i in k:
            group.append(int(i))
        clusters_B.append(group)
    
    A_hierarchial_word = []
    A_hierarchial_group = []
    for k in clusters_A:
        group =[]
        for i in k:
            group.append(A_list[i])
        A_hierarchial_group.append(group)
    
    B_hierarchial_word = []
    B_hierarchial_group = []
    for k in clusters_B:
        group = []
        for i in k:
            group.append(A_list[i])
        B_hierarchial_group.append(group)
    
    A_hierarchial_i = []
    a = 0 # muokkaa
    while a < 6:
#        if 11 < a and a < 22:
#           a += 1
#        else:
        for name in A_hierarchial_group[a]:
            A_hierarchial_word.append(name)
            i=0
            for row in A_vocab.loc[:,A_name]:
                if (name == cslb_vocab.loc[i,A_name]):
                    A_hierarchial_i.append(i)
                    break
                else:
                    i += 1
        a += 1
    A_hierarchial = A.iloc[A_hierarchial_i]
    
    B_hierarchial_i = []
    b = 0 # muokkaa
    while b < 3:
        for name in B_hierarchial_group[b]:
            B_hierarchial_word.append(name)
            i=0
            for row in B_vocab.loc[:,A_name]:
                if (name == B_vocab.loc[i,A_name]):
                    B_hierarchial_i.append(i)
                    break
                else:
                    i += 1
        b += 1
    B_hierarchial = B.iloc[B_hierarchial_i]
    return A_hierarchial,B_hierarchial,A_hierarchial_word,B_hierarchial_word,A_hierarchial_group,B_hierarchial_group
'''    