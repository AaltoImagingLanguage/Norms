import pandas as pd
import json
import os
metadata_fname = '/m/nbe/project/aaltonorms/data/{name}/metadata.tsv'
vocab_fname = '/m/nbe/project/aaltonorms/data/{name}/vocab.csv'
vectors_fname = '/m/nbe/project/aaltonorms/data/{name}/vectors.csv'
html_path = '../data'
url = 'https://raw.githubusercontent.com/AaltoImagingLanguage/SemanticNorms/master/data'
projector_config_fname = html_path + '/projector_config.json'

supernorms = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', 'Summary', index_col=0)
#subset = supernorms.dropna(subset=['aaltoprod', 'cslb', 'mcrae', 'vinson', 'cmu', 'w2v_fin', 'w2v_eng'])
subset = supernorms.dropna(subset=['aaltoprod', 'aalto85', 'w2v_fin', 'w2v_eng'])

projector_config = dict(embeddings=[])

#for name in ['aalto85', 'aaltoprod', 'cmu', 'cslb', 'vinson', 'w2v_eng', 'w2v_fin']:
for name in ['aalto85', 'aaltoprod', 'w2v_eng', 'w2v_fin']:
    vocab = pd.read_csv(vocab_fname.format(name=name), sep='\t', header=None)
    vocab.columns = [name]
    metadata = vocab.join(supernorms.set_index(name)[['fin_name', 'eng_name', 'category', 'word_class']],
                          on=name)
    metadata = metadata.drop_duplicates(name)
    metadata = metadata[['eng_name', 'fin_name', 'category', 'word_class']]
    metadata.to_csv(metadata_fname.format(name=name), sep='\t', index=False)
    vectors = pd.read_csv(vectors_fname.format(name=name), sep='\t', header=None)

    # Write data to public_html folder
    os.makedirs(html_path, exist_ok=True)
    os.makedirs(f'{html_path}/{name}', exist_ok=True)
    vectors.to_csv(f'{html_path}/{name}/vectors.tsv', sep='\t', header=False, index=False)
    metadata.to_csv(f'{html_path}/{name}/metadata.tsv', sep='\t', index=False)

    projector_config['embeddings'].append(
        dict(
            tensorName=name,
            tensorShape=vectors.shape,
            tensorPath=f'{url}/{name}/vectors.tsv',
            metadataPath=f'{url}/{name}/metadata.tsv',
        )
    )

    # Select common subset
    metadata_subset = metadata.reset_index().set_index('eng_name')
    metadata_subset = metadata_subset.loc[metadata_subset.index.intersection(subset['eng_name'])].reset_index()
    vectors_subset = vectors.iloc[metadata_subset['index']]
    del metadata_subset['index']

    vectors_subset.to_csv(f'{html_path}/{name}/vectors_subset.tsv', sep='\t', header=False, index=False)
    metadata_subset.to_csv(f'{html_path}/{name}/metadata_subset.tsv', sep='\t', index=False)

    projector_config['embeddings'].append(
        dict(
            tensorName=f'{name}-subset',
            tensorShape=vectors.shape,
            tensorPath=f'{url}/{name}/vectors_subset.tsv',
            metadataPath=f'{url}/{name}/metadata_subset.tsv',
        )
    )

with open(projector_config_fname, 'w') as f:
    json.dump(projector_config, f, indent=1)
