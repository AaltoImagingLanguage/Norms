# -*- coding: utf-8 -*-
"""
This script extracts different sets of semantic norms (Aalto, CSLB, Vinson, 
Corpus, Questionnaire) and compares them.

@author: kivisas1 (sasa@neuro.hut.fi)
Last update: 30.5.2018
"""
#from scipy.io import loadmat
import pandas
from scipy.io import loadmat
from scipy.spatial.distance import squareform, pdist
from matplotlib import pyplot as plt
import numpy as np

#import numpy as np
#import numpy as np
stimulus_list = '/m/nbe/scratch/aaltonorms/stimuli/aaltonorms_stimulus_set.csv'
cslb_norms_file  = '/m/nbe/scratch/guessfmri/aaltonorms/CSLB_norms/feature_matrix.dat'
aalto_norms_file = '/m/nbe/scratch/guessfmri/aaltonorms/AaltoNorms/lemma_sorted20151027_dl_synmerge.mat'
vinson_norms_file = '/m/nbe/scratch/guessfmri/aaltonorms/VinsonFeatures/VinsonFeature_Matrix_edit.csv'
aalto85_norms_file = '/m/nbe/scratch/guessfmri/aaltonorms/Aalto85questions/Aalto85_sorted20160204.mat'
figure_dir = '/m/nbe/scratch/aaltonorms/figures/'
ginter_norms_file = '/m/nbe/scratch/aaltonorms/ginter/AaltoNorm_words/lemma/context_5+5/ginter_lemma_5+5/concepts_vectors.csv'

#This is the list of stimuli in the Aalto production norms
aaltostimuli = pandas.read_table(
    stimulus_list,
    encoding='utf-8', header=None, index_col=False,
    names=['id', 'concept_eng', 'concept_fin', 'category', 'category_id', 
    'allnorms', 'cmu', 'cslb', 'vinson', 'aalto85', 'ginter']
)

aaltostimuli.sort_values(by='category_id') #Sort concepts by category
#Get CSLB names

cslb_names = aaltostimuli["cslb"]
vinson_names = aaltostimuli["vinson"]
orignames = aaltostimuli["concept_fin"]
ginter_names = aaltostimuli["ginter"]

#Aalto production data
fname = aalto_norms_file
m = loadmat(fname)
aaltowords = m['sorted']['word'][0][0]

#List Aaltonorms concepts
aalto_names = []
for i, w in enumerate(aaltowords):
    word = aaltowords[i][0][0][0][0]
    aalto_names.append(word)

#This is the full aaltonorms concept x feature matrix
#Dimensions: 300 (concepts) x 5683 (semantic features) 
aaltonorms = pandas.DataFrame(m['sorted']['mat'][0][0], index=aalto_names)

#Aalto85
fname = aalto85_norms_file
m = loadmat(fname)
aalto85words = m['sorted']['word'][0][0]

#List Aaltonorms 85concepts
aalto85_names = []
for i, w in enumerate(aalto85words):
    word = aalto85words[i][0][0]
    aalto85_names.append(word)    
aalto85norms = pandas.DataFrame(m['sorted']['mat'][0][0], index=aalto85_names)


#This is the CSLB full semantic concept x features matrix
#Dimensions: 638 (concepts) x 2725(semantic features) array
cslbnorms_orig = pandas.read_table(cslb_norms_file, encoding='latin1', 
                                   header=0, index_col=0)

#Get ginter vector labels
ginternorms_orig = pandas.read_table(ginter_norms_file, encoding='utf-8', 
                                     header=None , index_col=0)

#Vinson full semantic concept x features matrix
#Dimensions: 173 (concepts) x 1027(semantic features) array
vinsonnorms_orig = pandas.read_table(vinson_norms_file, encoding='latin1', 
                                     header=0, index_col=0, delimiter=",")
vinsonnorms_orig = vinsonnorms_orig.transpose() #Since this was the other way around originally
 
#Make aaltonorms dataframe with only those concepts that exist in the new norm
#set 
def select_aaltodata(names):
    df = pandas.DataFrame()
    for i, name in enumerate(names):
        if isinstance(names[i],unicode):
            origname = orignames[i] #Get corresponding aalto name
            df.name = origname
            data = aaltonorms.loc[[origname]] #Get corresponding row from aaltonorms
            df = df.append(data)
    return df

#Select concepts from a new norm set that exist in the aaltonorms 
def select_normdata(names, new_norms):
    df = pandas.DataFrame(columns=new_norms.columns.values)
    for i, name in enumerate(names):
        if isinstance(names[i],unicode):
            data = new_norms.loc[[name]]
            df = df.append(data)   
    return df
    
def get_distances(norms):
    distmat = squareform(pdist(norms, metric="cosine"))
    distvector = np.asarray(np.triu(distmat).reshape(-1)) #Take upper triangular
                                                            #and reshape
    return distmat, distvector
    

def compare_norms(A,B, label_A, label_B):
    plt.figure(figsize=(10,3))    
    ax1 = plt.subplot(1,2,1)
    plt.title(label_A)
    plt.imshow(get_distances(A)[0], cmap="plasma", interpolation='nearest')
    ax2 = plt.subplot(1,2,2)
    plt.title(label_B)
    plt.imshow(get_distances(B)[0], cmap="plasma", interpolation='nearest')
    plt.clim(0, 1);
    plt.colorbar(ax=[ax1, ax2])
    corr = np.corrcoef(get_distances(A)[1], get_distances(B)[1])
    print(label_A + " vs. " + label_B + " correlation coefficient is: " + str(round(corr[0][1],2)))
    plt.savefig(figure_dir + label_A + label_B + "_production_norm_comparison.pdf", format='pdf', dpi=1000, bbox_inches='tight')


#Make aaltonorms dataframe with only those concepts that exist in the CSLB/Vinson 
#dataset      
cslb_aaltonorms= select_aaltodata(cslb_names)
vinson_aaltonorms = select_aaltodata(vinson_names)
aalto85_aaltonorms =select_aaltodata(aalto85_names)
ginter_aaltonorms =select_aaltodata(ginter_names)

#Make vinsonnorms dataframe with only those concepts that exist in the aaltonorms 
cslbnorms = select_normdata(cslb_names, cslbnorms_orig)
vinsonnorms = select_normdata(vinson_names, vinsonnorms_orig)
ginternorms = select_normdata(ginter_names, ginternorms_orig)
#ginternorms = ginternorms.reindex(aaltonorms.index)


#Make distance matrices
compare_norms(cslb_aaltonorms, cslbnorms, "Aalto", "CSLB")
compare_norms(vinson_aaltonorms, vinsonnorms, "Aalto", "Vinson")
compare_norms(aalto85_aaltonorms, aalto85norms, "Aalto", "Aalto85")
compare_norms(ginter_aaltonorms, ginternorms, "Aalto", "Ginter")

