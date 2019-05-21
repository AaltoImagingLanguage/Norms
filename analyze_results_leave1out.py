import argparse

from scipy.io import loadmat
from scipy.misc import comb
import pandas as pd
import numpy as np

# Deal with command line arguments
parser = argparse.ArgumentParser(description='Analyze predictions made by the zero-shot learning and write the results to a CSV file.')
parser.add_argument('input_file', nargs='+', type=str,
                    help='The file(s) to use as input; should be (a) .mat file(s).')
parser.add_argument('-o', '--output', metavar='filename', type=str, default='/m/nbe/scratch/aaltonorms/results/zero_shot/noname_results.csv',
                    help='The filename to use to write the output; should end in ".csv". Defaults to /m/nbe/scratch/aaltonorms/results/zero_shot/noname_results.csv')
parser.add_argument('-s', '--stimuli', metavar='filename', type=str, default="/m/nbe/scratch/aaltonorms/data/item_category_list.csv",
                    help='The CSV file listing the stimuli; should end in ".csv". Defaults to /m/nbe/scratch/aaltonorms/data/item_category_list.csv')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print the stats as they are computed')
args = parser.parse_args()

stimuli = pd.read_csv(
    "/m/nbe/scratch/aaltonorms/data/item_category_list.csv",
    encoding='latin1', header=None, sep="\t", index_col=False,
    names=['word', 'category']
)
cateind = stimuli['category'].values




for i, fname in enumerate(args.input_file, 1):
    m = loadmat(fname, variable_names = ['accuracy', 'iteration', 'distance_matrix',
                                         'confusion_matrix'])
    num_words = len(cateind)
    dist = m['distance_matrix']
    #Checks which column index has the smallest distance to predicted and whether
    #it matches the predicted target index
    item_accuracy = np.mean(dist.argmin(axis=1) == np.arange(num_words))
    
    within_acc_list = []
    for i, test in enumerate(dist.argmin(axis=1)):
        predicted_category = cateind[test]
        true_category = cateind[i]
        if predicted_category == true_category:
            within_acc_list.append(1)
        else:
            within_acc_list.append(0)
    within_accuracy =  np.mean(within_acc_list)


results = {
        "correct_item" : [item_accuracy], 
        "correct_category" : [within_accuracy]
        }
results = pd.DataFrame.from_dict(results)
results.to_csv(args.output)

if args.verbose:
    print("Item-level accuracy:\t\t" + str(item_accuracy))
    print("Category-level accuracy:\t" + str(within_accuracy))