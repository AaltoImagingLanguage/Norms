"""
Analyze predictions made by the zero-shot learning and write the results to a
CSV file. The file contains the following columns:

subject:
The subject on which the analysis was run.

X and Y:
The categories that are being compared. For example, X='Animals' versus
Y='Tools'. For within category comparisons, X and Y are the same, for example
X='Animals' versus Y='Animals'. The X='all' versus Y='all' comparison compares
all words against all words (this would be the overall accuracy and the one you
are probably the most interested in at the moment). There is also X='category'
versus Y='within', which contains the overall within-category accuracy and
there is X='category' versus Y='between', which contains the overall
between-category accuracy.

accuracy:
The mean accuracy for the X versus Y comparison.

iteration:
For the random permutations, this column contains values 1-1000 which
correspond to the iteration which was run.

Author: Marijn van Vliet <marijn.vanvliet@aalto.fi>
"""

import argparse

from scipy.io import loadmat
from scipy.misc import comb
import pandas
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

# Read information about the stimuli
stimuli = pandas.read_csv(
    args.stimuli,
    encoding='latin1', header=None, sep="\t", index_col=False,
    names=['word', 'category']
)
cateind = stimuli['category'].values


def predictions_breakdown(accuracies, category_assignments, category_labels,
                          verbose=True):
    """
    Compute and display various statistics regarding the accuracy of the
    predictions made by the zero-shot learning approach.

    Parameters
    ---------
    accuracies: 2D array (n_examples, n_examples)
        The pairwise accuracy matrix.
    category_assignments : 1D array (n_examples,)
        For each example, the number of the category it belongs to.
    category_labels : dict (int -> str)
        A dictionary mapping each category number to a string label.
    verbose : bool
        Whether to print out the results as they are computed.
        Defaults to True.

    Returns
    -------
    results : instance of DataFrame
        A Pandas DataFrame object that contains the results.

    See also
    --------
    zero_shot_decoding
    """
    # Sanity checks
    assert accuracies.shape[0] == accuracies.shape[1]

    n_examples = accuracies.shape[0]
    assert len(category_assignments) == n_examples, (
        'number of rows in accuracies should match the length of '
        'category_assignments')

    category_codes = np.unique(category_assignments)
    n_categories = len(category_codes)

    # Results are collected in this list as (key, value) tuples.
    # (keys are (str, str) tuples themselves)
    results = []

    if verbose:
        print('\nAccuracy\t#Comp.\tDescription')
        print('-----------------------------------')

    mask = np.triu(np.ones_like(accuracies, dtype=np.bool), 1)
    accuracy = np.mean(accuracies[mask])
    n_comparisons = len(accuracies[mask])
    results.append((('all', 'all'), (accuracy, n_comparisons)))
    if verbose:
        print('%f\t%d\t%s' % (accuracy, n_comparisons, 'Overall accuracy'))

    # Within category accuracies
    within_cat_accuracies = np.zeros(n_categories)
    n_within_comparisons = np.zeros(n_categories)
    for i, category_code in enumerate(category_codes):
        words_in_cat = category_assignments == category_code
        trimmed_accuracies = accuracies[words_in_cat, :][:, words_in_cat]
        mask = np.triu(np.ones_like(trimmed_accuracies, dtype=np.bool), 1)
        within_cat_accuracies[i] = np.mean(trimmed_accuracies[mask])
        n_within_comparisons[i] = len(trimmed_accuracies[mask])
        accuracy = within_cat_accuracies[i]
        results.append((
            (category_labels[category_code], category_labels[category_code]),
            (accuracy, n_within_comparisons[i])
        ))
        if verbose:
            print('%f\t%d\tWithin %s' % (accuracy,
                                         n_within_comparisons[i],
                                         category_labels[category_code]))

    # Between category accuracies
    between_cat_accuracies = np.zeros(int(comb(n_categories, 2)))
    n_between_comparisons = np.zeros(int(comb(n_categories, 2)))
    ind = 0
    for i in range(n_categories):
        for j in range(i + 1, n_categories):
            words_in_cat1 = category_assignments == category_codes[i]
            words_in_cat2 = category_assignments == category_codes[j]
            mask = np.zeros_like(accuracies, dtype=np.bool)
            mask[np.dot(
                words_in_cat1[:, np.newaxis],
                words_in_cat2[np.newaxis, :]
            )] = True
            mask = np.triu(mask, 1)
            between_cat_accuracies[ind] = np.mean(accuracies[mask])
            n_between_comparisons[ind] = len(accuracies[mask])
            accuracy = between_cat_accuracies[ind]
            results.append((
                (category_labels[category_codes[i]],
                 category_labels[category_codes[j]]),
                (accuracy, n_between_comparisons[ind])
            ))
            if verbose:
                print('%f\t%d\t%s vs %s' % (accuracy,
                                            n_between_comparisons[ind],
                                            category_labels[category_codes[i]],
                                            category_labels[category_codes[j]]))
            ind = ind + 1

    overall_within_cat_accuracy = (
        np.sum(within_cat_accuracies * n_within_comparisons) /
        np.sum(n_within_comparisons)
    )
    overall_between_cat_accuracy = (
        np.sum(between_cat_accuracies * n_between_comparisons) /
        np.sum(n_between_comparisons)
    )
    results.append((('category', 'within'),
                    (overall_within_cat_accuracy, np.sum(n_within_comparisons))))
    results.append((('category', 'between'),
                    (overall_between_cat_accuracy, np.sum(n_between_comparisons))))

    if verbose:
        print('%f\t%d\tOverall within category' % (
            overall_within_cat_accuracy, np.sum(n_within_comparisons)))
        print('%f\t%d\tOverall between category' % (
            overall_between_cat_accuracy, np.sum(n_between_comparisons)))

    return pandas.DataFrame(
        index=pandas.MultiIndex.from_tuples([x[0] for x in results],
                                            names=['X', 'Y']),
        data=[x[1] for x in results],
        columns=['accuracy', 'n_comparisons'],
    )

# Obtain results for all input files
results = []
for i, fname in enumerate(args.input_file, 1):
    m = loadmat(fname, variable_names = ['accuracy', 'iteration', 'confusion_matrix'])
    A = m['confusion_matrix']

    category_labels = {
		1 : "animal" ,
		2 : "bodypart",
		3 : "building",
		4 : "clothing",
		5 : "container",
		6 : "fruit",
        7 : "furniture",
        8 : "tool",
        9 : "vegetable",
        10 : "vehicle",
        11 :"weapon"       
	}


    df = predictions_breakdown(A, cateind, category_labels, verbose=args.verbose)
    #df['subject'] = m['subject'][0][0]

    # Results from random permutations have the 'iteration' field set
    if 'iteration' in m:
        df['iteration'] = i

    results.append(df.reset_index())

# Collect all the results in one big table
results = pandas.concat(results, ignore_index=True)

# Set the proper index, based on whether we are analyzing real data or random
# permutations.
if 'iteration' in results:
    results = results.set_index(['iteration', 'X', 'Y'])
#else:
    #results = results.set_index(['subject', 'X', 'Y'])

# Save the table
results.to_csv(args.output)
