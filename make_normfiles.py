"""
This script extracts different sets of semantic norms (Aalto, CSLB, Vinson, 
Corpus, Questionnaire) and compares them.

@author: kivisas1 (sasa@neuro.hut.fi)

"""
#from scipy.io import loadmat
import pandas
from scipy.io import loadmat


norms_path = '/m/nbe/project/aaltonorms/data/'
out_path = '/m/nbe/project/aaltonorms/data/norm_overlap/'

#These are the files for different norm sets
cslb_norms_file  = norms_path + 'CSLB/feature_matrix.dat'
aalto300_file = norms_path + 'AaltoProduction/lemma_sorted20151027_dl_synmerge.mat'
vinson_norms_file = norms_path + 'Vinson/VinsonFeature_Matrix_edit.csv'
aalto85_norms_file = norms_path + 'Aalto85questions/Aalto85_sorted20160204.mat'
ginter_norms_file = norms_path + 'Ginter/ginter-300-5+5/AaltoNorm_words/lemma/context_5+5/ginter_lemma_5+5/concepts_vectors.csv'
cmu_norms_file = norms_path + 'CMU/bagOfFeatures.mat'
ginter_cslb_norms_file = norms_path + 'Ginter/ginter-300-5+5/cslb/concepts_vectors.csv'
ginter_vinson_norms_file = norms_path + 'Ginter/ginter-300-5+5/vinson/concepts_vectors.csv'

#Name correspondence files
stimulus_list = '/m/nbe/project/aaltonorms/stimulus_LUT/aaltonorms_stimulus_set.csv'
cslb_to_ginter = norms_path + 'CSLB/cslb_ginter_correspondence.csv'
vinson_to_ginter = norms_path + 'Vinson/vinson_ginter_correspondence.csv'


#Directory for the figures to be created
figure_dir = '/m/nbe/project/aaltonorms/results/figures'

#This is the list of Aalto production norms stimuli and the corresponding names
#for the same stimuli in other norm sets
aaltostimuli = pandas.read_table(
    stimulus_list, encoding='utf-8', header=None, index_col=2,
    names=['id', 'concept_eng', 'concept_fin', 'category', 'category_id', 
    'allnorms', 'cmu', 'cslb', 'vinson', 'aalto85', 'ginter']
)

#Corresponcence of CSLB names and Ginter translations
cslb_ginter_names = pandas.read_table(
    cslb_to_ginter, encoding='utf-8', header=None, index_col=0,
    delimiter = ",", names=['cslb', 'ginter']
)

#Corresponcence of Vinson names and Ginter translations
vinson_ginter_names = pandas.read_table(
    vinson_to_ginter, encoding='utf-8', header=None, index_col=0,
    delimiter = ";", names=['vinson', 'ginter']
)

aaltostimuli.sort_values(by='category_id') #Sort concepts by category

#Get list of concept names
cslb_names = aaltostimuli["cslb"]
vinson_names = aaltostimuli["vinson"]
ginter_names = aaltostimuli["ginter"]
cmu_names = aaltostimuli["cmu"]
category = aaltostimuli ["category_id"]

#Extract aalto production norm data from a Matlab file
def get_matlab_arrays(norm_file):
    fname =norm_file
    m = loadmat(fname)
    vectorarray = pandas.DataFrame(m['sorted']['mat'][0][0])
    wordarray = m['sorted']['word'][0][0]
    return vectorarray, wordarray


#Make aalto300 dataframe with only those concepts that exist in the new norm
#set 
def select_aaltodata(name_list, norm):
    df = pandas.DataFrame()
    cats = pandas.DataFrame()
    for i, name in enumerate(name_list):
        if isinstance(name_list[i], str):
            origname =aaltostimuli.loc[aaltostimuli[norm] == name].index.tolist()[0]
            #cat = aaltostimuli.loc[[origname],['category_id']]
            #print(cat)
            #cats = cats.append(cat.values[0][0])
            df.name = origname
            data = aalto300.loc[[origname]] #Get corresponding row from aalto300
            df = df.append(data)
    
    return df, cats

#Select concepts from a new norm set that exist in the aalto300 
def select_normdata(names, new_norms):
    df = pandas.DataFrame(columns=new_norms.columns.values)
    for i, name in enumerate(names):
        if isinstance(names[i], str):
            data = new_norms.loc[[name]]
            df = df.append(data)   
    return df


#This function selects corresonding feature vectors from ginter norms and 
#some type of production norms. Name correspondence is required.
def select_normdata_ginter (norms, names):
    df1 = pandas.DataFrame()
    df2 = pandas.DataFrame()
    for name in names.index.values:

        gintername = names.loc[[name]].values[0][0]
        try:
            #Check if name found in both files
            data1 = norms.loc[[name]]
            data2 = ginter_all.loc[[gintername]]
            
            #If it is found append the new dataframes
            df1.name = name
            df2.name = gintername
            df1 = df1.append(data1)
            df2 = df2.append(data2)
        except:
           print("The concept " + name + " is not in the ginterlist." )        
        
    return df1, df2


aaltowordarray = get_matlab_arrays(aalto300_file)[1]
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


#This is the full aalto300 concept x feature matrix
#Dimensions: 300 (concepts) x 5683 (semantic features) 
aalto300 = pandas.DataFrame(data = get_matlab_arrays(aalto300_file)[0].values, 
                              index=aalto_names)

#Aalto85 norm concept x feature matrix
aalto85_aaltoOverlap = pandas.DataFrame(data = get_matlab_arrays(aalto85_norms_file)[0].values, 
                                index=aalto85_names)

#This is the CSLB semantic concept x features matrix
#Dimensions: 638 (concepts) x 2725(semantic features) array
cslbnorms_orig = pandas.read_table(cslb_norms_file, encoding='latin1', 
                                   header=0, index_col=0)

#Get ginter vector matrix (context 5+5, lemma)
ginternorms_orig = pandas.read_table(ginter_norms_file, encoding='utf-8', 
                                     header=None , index_col=0)

ginter_cslb_norms_orig = pandas.read_table(ginter_cslb_norms_file, encoding='utf-8', 
                                     header=None , index_col=0, delimiter=";")

ginter_vinson_norms_orig = pandas.read_table(ginter_vinson_norms_file, encoding='utf-8', 
                                     header=None , index_col=0, delimiter=";")


#Make big ginter data frame using only unique concept names
ginter_all = ginternorms_orig
for item in ginter_cslb_norms_orig.index.values:    
    try:
        ginter_all.loc[[item]]
    except:
        ginter_all = ginter_all.append(ginter_cslb_norms_orig.loc[[item]])
        
for item in ginter_vinson_norms_orig.index.values:    
    try:
        ginter_all.loc[[item]]
    except:
        ginter_all = ginter_all.append(ginter_vinson_norms_orig.loc[[item]])
ginter_all = ginter_all.sort_index()
     

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


#Make aalto300 dataframe with only those concepts that exist in the CSLB/Vinson 
#dataset      
aalto300_cslbOverlap, cslb_catlist = select_aaltodata(cslb_names, norm = 'cslb')
aalto300_vinsonOverlap, vinson_catlist = select_aaltodata(vinson_names, 'vinson')
aalto300_aalto85Overlap, aalto85_catlist = select_aaltodata(aalto85_names, norm = 'aalto85')
aalto300, ginter_catlist = select_aaltodata(ginter_names, norm = 'ginter')
aalto300_cmuOverlap, cmu_catlist = select_aaltodata(cmu_names, norm = 'cmu')


#Make dataframes with only those concepts that exist in the aalto300 
cslb_aaltoOverlap = select_normdata(cslb_names, cslbnorms_orig)
vinson_aaltoOverlap = select_normdata(vinson_names, vinsonnorms_orig)
ginter_aaltoOverlap = select_normdata(ginter_names, ginternorms_orig)
cmu_aaltoOverlap = select_normdata(cmu_names, cmunorms_orig)

##Select items that overlap in the ginter data
cslb_ginterOverlap, ginter_cslbOverlap = select_normdata_ginter(cslbnorms_orig, cslb_ginter_names)
vinson_ginterOverlap, ginter_vinsonOverlap = select_normdata_ginter(vinsonnorms_orig, vinson_ginter_names)

##List of dataframes
normfiles = [aalto300_cslbOverlap, aalto300_vinsonOverlap, aalto300_aalto85Overlap,
aalto300, aalto300_cmuOverlap, cslb_aaltoOverlap, aalto85_aaltoOverlap, vinson_aaltoOverlap, 
ginter_aaltoOverlap, cmu_aaltoOverlap, cslb_ginterOverlap, 
vinson_ginterOverlap, ginter_cslbOverlap, ginter_vinsonOverlap]

#Dataframe names
normnames = ['aalto300_cslbOverlap', 'aalto300_vinsonOverlap', 'aalto300_aalto85Overlap',
'aalto300', 'aalto300_cmuOverlap', 'cslb_aaltoOverlap', 'aalto85_aaltoOverlap', 'vinson_aaltoOverlap', 
'ginter_aaltoOverlap', 'cmu_aaltoOverlap', 'cslb_ginterOverlap', 
'vinson_ginterOverlap', 'ginter_cslbOverlap', 'ginter_vinsonOverlap']

#Save norms to files

for i, normfile in enumerate(normfiles):
    
    normfile.to_csv(out_path + normnames[i] + '.csv', header=False, index=True, sep='\t', encoding='utf-8')
 
