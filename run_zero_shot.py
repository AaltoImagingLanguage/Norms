# encoding: utf-8

"""
This script runs the zero shot learning algorithm using preselected set of voxels (save_top_voxels.py)
"""

from scipy.io import savemat
from zero_shot_decoding import zero_shot_decoding
#import subprocess

def run_zero_shot(norms1, norms2, distance_metric, output):
    #Load the first set of norm data
    norms1 = norms1.loc[:, (norms1 != 0).any(axis=0)] #Drop columns where all values = 0
    X = norms1.values
    # Load the second set of norm data
    norms2 = norms2.loc[:, (norms2 != 0).any(axis=0)] #Drop columns where all values = 0
    y = norms2.values
    
    
    pairwise_accuracies, model, target_scores, predicted_y, patterns = zero_shot_decoding(
        X, y, verbose=True, metric=distance_metric
    )
    
    savemat(output, {
        'pairwise_accuracies': pairwise_accuracies,
        'weights': model.coef_,
        'feat_scores': target_scores,
        'alphas': model.alpha_,
        'predicted_y': predicted_y,
        'patterns': patterns,
    })
