import pandas as pd
import json
import os
metadata_fname = '/m/nbe/project/aaltonorms/data/{name}/metadata.tsv'
vocab_fname = '/m/nbe/project/aaltonorms/data/{name}/vocab.csv'
vectors_fname = '/m/nbe/project/aaltonorms/data/{name}/vectors.csv'
html_path = '../data'
url = 'https://users.aalto.fi/~vanvlm1/aaltonorms'
#url = 'https://raw.githubusercontent.com/wmvanvliet/aaltonorms/master'
projector_config_fname = html_path + '/projector_config.json'

supernorms = pd.read_excel('/m/nbe/project/aaltonorms/data/SuperNormList.xls', 'Summary', index_col=0)

projector_config = dict(embeddings=[])

for name in ['aalto85', 'aaltoprod', 'cmu', 'cslb', 'vinson', 'w2v_eng', 'w2v_fin']:
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

with open(projector_config_fname, 'w') as f:
    json.dump(projector_config, f, indent=1)
