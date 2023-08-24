import argparse

from scipy.io import loadmat
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



results = []
for i, fname in enumerate(args.input_file, 1):
    m = loadmat(fname)
    num_words = len(cateind)
    dist = m['distance_matrix']
    #Checks which column index has the smallest distance to predicted and whether
    #it matches the predicted target index
    item_accuracy = np.mean(dist.argmin(axis=1) == np.arange(num_words))

    within_acc_list = []
    for j, test in enumerate(dist.argmin(axis=1)):
        predicted_category = cateind[test]
        true_category = cateind[j]

        if predicted_category == true_category:
            within_acc_list.append(1)
        else:
            within_acc_list.append(0)
            within_accuracy =  np.mean(within_acc_list)

    if 'iteration' in m:
        results.append([m['iteration'][0][0], item_accuracy, within_accuracy])
    else:
        results.append([item_accuracy, within_accuracy])


# Collect all the results in one big table
if 'iteration' in m:
    results = pd.DataFrame(results, columns=["iteration", "item-level", "category-level"])
else:
    results = pd.DataFrame(results, columns=["item-level", "category-level"])
# Set the proper index, based on whether we are analyzing real data or random
# permutations.
#if 'iteration' in results:
#    results = results.set_index(['iteration', 'X', 'Y'])

results.to_csv(args.output)

if args.verbose:
    print("Item-level accuracy:\t\t" + str(item_accuracy))
    print("Category-level accuracy:\t" + str(within_accuracy))
