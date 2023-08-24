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
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print the stats as they are computed')
args = parser.parse_args()



results = []
for i, fname in enumerate(args.input_file, 1):
    m = loadmat(fname)
    words = [w[0] for w in m['words'][0]]
    categories = [c[0] for c in m['categories'][0]]
    num_words = len(words)
    dist = m['distance_matrix']
    #Checks which column index has the smallest distance to predicted and whether
    #it matches the predicted target index
    item_accuracy = np.mean(dist.argmin(axis=1) == np.arange(num_words))

    within_acc_list = []
    for j, test in enumerate(dist.argmin(axis=1)):
        predicted_category = categories[test]
        true_category = categories[j]

        if predicted_category == true_category:
            within_acc_list.append(1)
        else:
            within_acc_list.append(0)
            within_accuracy = np.mean(within_acc_list)

    if 'iteration' in m:
        results.append([m['iteration'][0][0], item_accuracy, within_accuracy])
    else:
        results.append([item_accuracy, within_accuracy])


# Collect all the results in one big table
if 'iteration' in m:
    results = pd.DataFrame(results, columns=["iteration", "item-level", "category-level"])
else:
    results = pd.DataFrame(results, columns=["item-level", "category-level"])

results.to_csv(args.output)

if args.verbose:
    item_level_thres = results['item-level'].sort_values().iat[int(0.95 * len(results))]
    cat_level_thres = results['category-level'].sort_values().iat[int(0.95 * len(results))]
    print(f"Item-level accuracy:\t\t{results['item-level'].mean()} (p=0.05: {item_level_thres})")
    print(f"Category-level accuracy:\t{results['category-level'].mean()} (p=0.05: {cat_level_thres})")
