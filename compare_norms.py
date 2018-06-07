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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr

#These are the files for different norm sets
stimulus_list = '/m/nbe/scratch/aaltonorms/stimuli/aaltonorms_stimulus_set.csv'
cslb_norms_file  = '/m/nbe/scratch/aaltonorms/CSLB_norms/feature_matrix.dat'
aalto_norms_file = '/m/nbe/scratch/aaltonorms/AaltoNorms/lemma_sorted20151027_dl_synmerge.mat'
vinson_norms_file = '/m/nbe/scratch/aaltonorms/VinsonFeatures/VinsonFeature_Matrix_edit.csv'
aalto85_norms_file = '/m/nbe/scratch/aaltonorms/Aalto85questions/Aalto85_sorted20160204.mat'
ginter_norms_file = '/m/nbe/scratch/aaltonorms/ginter/AaltoNorm_words/lemma/context_5+5/ginter_lemma_5+5/concepts_vectors.csv'
cmu_norms_file = '/m/nbe/scratch/aaltonorms/CMU_norms/bagOfFeatures.mat'


#Output data file names
cslb_out_file = '/m/nbe/scratch/aaltonorms/norms/cslb_norms_aalto_overlap.pkl'
vinson_out_file = '/m/nbe/scratch/aaltonorms/norms/vinson_norms_aalto_overlap.pkl'
aalto85_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto85_norms_aalto_overlap.pkl'
ginter_out_file = '/m/nbe/scratch/aaltonorms/norms/ginter_norms_aalto_overlap.pkl'
cmu_out_file = '/m/nbe/scratch/aaltonorms/norms/cmu_norms_aalto_overlap.pkl'
aalto_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_all.pkl'
aaltonorms_cslb_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_cslb_overlap.pkl'
aaltonorms_vinson_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_vinson_overlap.pkl'
aaltonorms_aalto85_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_aalto85_overlap.pkl'
aaltonorms_cmu_out_file = '/m/nbe/scratch/aaltonorms/norms/aalto_norms_cmu_overlap.pkl'


#Directory for the figures to be created
figure_dir = '/m/nbe/scratch/aaltonorms/figures/'

#This is the list of Aalto production norms stimuli and the corresponding names
#for the same stimuli in other norm sets
aaltostimuli = pandas.read_table(
    stimulus_list,
    encoding='utf-8', header=None, index_col=2,
    names=['id', 'concept_eng', 'concept_fin', 'category', 'category_id', 
    'allnorms', 'cmu', 'cslb', 'vinson', 'aalto85', 'ginter']
)
aaltostimuli.sort_values(by='category_id') #Sort concepts by category

#Get list of concept names
cslb_names = aaltostimuli["cslb"]
vinson_names = aaltostimuli["vinson"]
orignames = aaltostimuli["concept_fin"]
ginter_names = aaltostimuli["ginter"]
cmu_names = aaltostimuli["cmu"]
category = aaltostimuli ["category_id"]
#aalto85_names2 = aaltostimuli ["aalto85"] #To get category labels

#Extract aalto production norm data from a Matlab file
def get_matlab_arrays(norm_file):
    fname =norm_file
    m = loadmat(fname)
    vectorarray = pandas.DataFrame(m['sorted']['mat'][0][0])
    wordarray = m['sorted']['word'][0][0]
    return vectorarray, wordarray


#Make aaltonorms dataframe with only those concepts that exist in the new norm
#set 
def select_aaltodata(names):
    df = pandas.DataFrame()
    cats = pandas.DataFrame()
    for i, name in enumerate(names):
        if isinstance(names[i],unicode):
            origname = orignames[i] #Get corresponding aalto name
            cat = aaltostimuli.loc[[origname],['category_id']]
            cats = cats.append(cat)
            df.name = origname
            data = aaltonorms.loc[[origname]] #Get corresponding row from aaltonorms
            df = df.append(data)
    
    return df, cats

#Select concepts from a new norm set that exist in the aaltonorms 
def select_normdata(names, new_norms):
    df = pandas.DataFrame(columns=new_norms.columns.values)
    for i, name in enumerate(names):
        if isinstance(names[i],unicode):
            data = new_norms.loc[[name]]
            df = df.append(data)   
    return df


def select_normdata_notAalto (norm1, norm2):
    df1 = pandas.DataFrame()
    df2 = pandas.DataFrame()
    for i, name1 in enumerate(norm1.index.values):
        for j, name2 in enumerate(norm2.index.values):
            if name1 == name2:
                df1.name = name1
                df1.append()
                df2.name = name2
                df2 = df2.append(norm2.values[j])

def get_distances(norms):
    distmat = squareform(pdist(norms, metric="cosine"))
    distvector = np.asarray(np.triu(distmat).reshape(-1)) #Take upper triangular
                                                            #and reshape
    return distmat, distvector

def make_category_bar(cats):
    plt.figure(figsize=(1,3))    
    ax = plt.imshow(cats, cmap='Paired', interpolation='nearest', extent=[0,5,0,1], aspect=100)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticklabels([])
    return ax
    
def compare_norms(A,B, label_A, label_B, cats=None):
    plt.figure(figsize=(10,3))    
    ax1 = plt.subplot(1,2,1)
    plt.title(label_A)
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_xaxis().set_ticklabels([])
    ax1.axes.get_yaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticklabels([])
    plt.imshow(get_distances(A)[0], cmap="plasma", interpolation='nearest')
    ax2 = plt.subplot(1,2,2)
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_xaxis().set_ticklabels([])
    ax2.axes.get_yaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticklabels([])
    plt.title(label_B)
    plt.imshow(get_distances(B)[0], cmap="plasma", interpolation='nearest')
    plt.clim(0, 1);
    plt.colorbar(ax=[ax1, ax2])
   # plt.tight_layout()
    rho, pval = spearmanr(get_distances(A)[1], get_distances(B)[1])
    print(label_A + " vs. " + label_B + "  rho is: " + 
        str(round(rho,2)) + ", pvalue = " + str(pval))
    #plt.savefig(figure_dir + label_A + label_B + "_production_norm_comparison.pdf", 
    #            format='pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(figure_dir + label_A + label_B + "_production_norm_comparison.pdf", 
                format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()
    make_category_bar(cats)
    plt.savefig(figure_dir + label_A + label_B + "_catcolors.pdf", 
                format='pdf', dpi=1000, bbox_inches='tight')
        



def hierarchical_clustering(norms, truncate=True):
    mat = get_distances(norms)[0]
    labels = norms.index.values
    Z = linkage(mat, 'ward')
    plt.figure(figsize=(30, 10))
    if truncate:
        dendrogram(Z, labels=labels, p=5, truncate_mode='level', 
                   orientation='right', leaf_font_size=5)
    else: 
        dendrogram(Z, labels=labels, orientation='right', leaf_font_size=5)


aaltowordarray = get_matlab_arrays(aalto_norms_file)[1]
aalto85wordarray = get_matlab_arrays(aalto85_norms_file)[1]

#List concept names from the matlab cell array
aalto_names = []
for i, w in enumerate(aaltowordarray):
    word = aaltowordarray[i][0][0][0][0]
    aalto_names.append(word)


aalto85_names = []
for i, w in enumerate(aalto85wordarray):
    word = aalto85wordarray[i][0][0]
    aalto85_names.append(word)   


#This is the full aaltonorms concept x feature matrix
#Dimensions: 300 (concepts) x 5683 (semantic features) 
aaltonorms = pandas.DataFrame(data = get_matlab_arrays(aalto_norms_file)[0].values, 
                              index=aalto_names)

#Aalto85 norm concept x feature matrix
aalto85norms = pandas.DataFrame(data = get_matlab_arrays(aalto85_norms_file)[0].values, 
                                index=aalto85_names)


#This is the CSLB semantic concept x features matrix
#Dimensions: 638 (concepts) x 2725(semantic features) array
cslbnorms_orig = pandas.read_table(cslb_norms_file, encoding='latin1', 
                                   header=0, index_col=0)

#Get ginter vector matrix (context 5+5, lemma)
ginternorms_orig = pandas.read_table(ginter_norms_file, encoding='utf-8', 
                                     header=None , index_col=0)

#Vinson full semantic concept x features matrix
#Dimensions: 173 (concepts) x 1027(semantic features) array
vinsonnorms_orig = pandas.read_table(vinson_norms_file, encoding='latin1', 
                                     header=0, index_col=0, delimiter=",")
vinsonnorms_orig = vinsonnorms_orig.transpose() #Since this was the other way around originally

#Get CMU bag of features data.
#Dimensions: 1000 (concepts x 2xx (features)
fname =cmu_norms_file
m = loadmat(fname)
cmu_vectorarray = pandas.DataFrame(m['features'])
cmu_wordarray = m['nouns']

cmu_names_all = []
for i, w in enumerate(cmu_wordarray):
    word = cmu_wordarray[i][0][0]
    cmu_names_all.append(word)

cmunorms_orig = pandas.DataFrame(data = cmu_vectorarray.values, 
                                 index=cmu_names_all)


#Make aaltonorms dataframe with only those concepts that exist in the CSLB/Vinson 
#dataset      
cslb_aaltonorms, cslb_catlist = select_aaltodata(cslb_names)
vinson_aaltonorms, vinson_catlist = select_aaltodata(vinson_names)
aalto85_aaltonorms, aalto85_catlist = select_aaltodata(aalto85_names)
ginter_aaltonorms, ginter_catlist = select_aaltodata(ginter_names)
cmu_aaltonorms, cmu_catlist = select_aaltodata(cmu_names)


#Make vinsonnorms dataframe with only those concepts that exist in the aaltonorms 
cslbnorms = select_normdata(cslb_names, cslbnorms_orig)
vinsonnorms = select_normdata(vinson_names, vinsonnorms_orig)
ginternorms = select_normdata(ginter_names, ginternorms_orig)
cmunorms = select_normdata(cmu_names, cmunorms_orig)


#Make distance matrices
compare_norms(cslb_aaltonorms, cslbnorms, "Aalto", "CSLB", cslb_catlist)
compare_norms(vinson_aaltonorms, vinsonnorms, "Aalto", "Vinson", vinson_catlist)
compare_norms(aalto85_aaltonorms, aalto85norms, "Aalto", "Aalto85", aalto85_catlist)
compare_norms(ginter_aaltonorms, ginternorms, "Aalto", "Ginter", ginter_catlist)
compare_norms(cmu_aaltonorms, cmunorms, "Aalto", "CMU", cmu_catlist)

hierarchical_clustering(ginter_aaltonorms, truncate=False)

#Save norms to files
#cslb_aaltonorms.to_pickle(aaltonorms_cslb_out_file)
#vinson_aaltonorms.to_pickle(aaltonorms_vinson_out_file)
#aalto85norms.to_pickle(aalto85_out_file)
#aalto85_aaltonorms.to_pickle(aaltonorms_aalto85_out_file)
#ginter_aaltonorms.to_pickle(aalto_out_file) 
#cmu_aaltonorms.to_pickle(aaltonorms_cmu_out_file)
#cslbnorms.to_pickle(cslb_out_file) 
#vinsonnorms.to_pickle(vinson_out_file) 
#ginternorms.to_pickle(ginter_out_file) 
#cmunorms.to_pickle(cmu_out_file) 
