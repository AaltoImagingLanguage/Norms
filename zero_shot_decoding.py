#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform Zero-shot decoding.

In zero-shot learning, the goal is to learn a classifier f : X → Y that must
predict novel values of Y that were omitted from the training set. To achieve
this, we define the notion of a semantic output code classifier (SOC) which
utilizes a knowledge base of semantic properties of Y to extrapolate to novel
classes.

In other words:
Given a semantic encoding of a large set of concept classes, can we build a
classifier to recognize classes that were omitted from the training set?

References
----------
Palatucci, M., Hinton, G., Pomerleau, D., Mitchell, T. M. (2009). Zero-shot
learning with semantic output codes. In Y. Bengio, D. Schuurmans, J. D.
Lafferty, C. K. I. Williams, & A. Culotta (Eds.), Advances in Neural
Information Processing Systems 22 (pp. 1410–1418). Curran Associates, Inc.

"""


import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.cross_validation import LeavePOut

from ridge import RidgeGCV

def _leave_two_out_accuracy(ground_truth, predictions, metric='sqeuclidean'):
    """Compute classification accuracy given two examples.

    The classification task is to tell which example is which.

    A distance metric is used to determine which set of predicted targets best
    fits the ground truth of example 1 and which best fits the ground truth of
    example 2. The accuracy can therefore be either 100% or 0%.

    Parameters
    ----------
    ground_truth : 1D numpy array (2,) | 2D numpy array (2, n_targets)
        The real targets of the two examples.
    predictions : 1D numpy array (2,) | 2D numpy array (2, n_targets)
        The predicted targets for the two examples.
    metric : str
        The distance metric to use. Any distance implemented in SciPy's
        spatial.distance module is supported. See the docstring of
        scipy.spatial.distance.pdict for the exhaustive list of possitble
        metrics. Here are some of the more useful ones:
        'euclidean' - Euclidean distance
        'sqeuclidean' - Squared euclidean distance (the default)
        'correlation' - Pearson correlation
        'cosine' - Cosine similarity

    Returns
    -------
    accuracy : float
        Either 1 (correct) or 0 (incorrect).
    """
    assert len(ground_truth) == 2 and len(predictions) == 2

    # Make sure these are 2D matrices
    if ground_truth.ndim == 1:
        ground_truth = ground_truth[:, np.newaxis]
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]

    # Compute distances
    D = cdist(predictions, ground_truth, metric=metric)

    # Do the predicted targets match the ground truth targets?
    if (D[0, 0] + D[1, 1]) < (D[0, 1] + D[1, 0]):
        return 1.
    else:
        return 0.
    
def _patterns_from_weights(X, W, y_hat, alpha):
    """Compute activity patterns from regression weights.

    Regression weights originating from multivariage regression cannot be
    directly interpreted. Instead, they must be multiplied with the covariance
    matrix of the data in order to yield 'patterns' instead of 'filters' [1].
    The resulting patterns can be interpreted.

    Parameters
    ----------
    X : 2D array (n_samples, n_features)
        The data the learning was performed on.
    W : 1D array (n_features)
        The regression weights (may include the 1's added for learning the bias).
    y_hat : 2D array (n_samples, n_targets)
        The targets that where predicted from the data (not the ground thruth!).
    alpha : 1D array (n_targets)
        The optimal shrinkage parameters chosen by the regression algorithm.

    Returns
    -------
    P : 2D array (n_samples, n_features)
        The activity patterns.

    References
    ----------
    [1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D.,
    Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight
    vectors of linear models in multivariate neuroimaging. NeuroImage, 87,
    96–110.  http://doi.org/10.1016/j.neuroimage.2013.10.067
    """
    # Check input dimensions
    n_samples, n_features = X.shape
    if W.shape[1] != n_features:
        raise ValueError("The number of weights doesn't match the number of features.")
    assert y_hat.shape[0] == n_samples
    n_targets = y_hat.shape[1]
    assert len(alpha) == n_targets

    # Compute inverse covariance of the predicted targets
    SS = (y_hat - y_hat.mean(axis=0)) / np.sqrt(n_samples - 1)
    S_cov_i = np.linalg.pinv(SS.T.dot(SS))

    # Used shrinkage parameters
    alpha = alpha.mean()  # TODO separate alpha for each target

    # Compute patterns
    XX = (X - X.mean(axis=0)) / np.sqrt(n_samples - 1)


    P = XX.T.dot(XX.dot(W.T)).dot(S_cov_i).T
    return P

    
def zero_shot_decoding(X, y, verbose=True, normalize=True, model=None, 
                       metric='cosine', cv_splits=None):
    """Perform zero shot decoding.

    Zero-shot learning refers to a specific way to evaluate a model. Given two
    examples (rows of X) and two labels (rows of y), the task is to match each
    example to the right label. Note that this is a binary choice: when
    example 1 belongs to label 1, example 2 necessarily belongs to label 2;
    or vice versa.

    This function trains and evaluates a given model (which defaults to
    linear ridge regression) on each possible pair of samples, using a
    leave-two-out cross validation scheme.

    Parameters
    ----------
    X: 2D array (n_iterms, n_normfeatures1)
        The input data; the thing you want to make predictions from. In this
        project it contains the a reference norm set
    y : 2D array (n_items, n_normfeatures2)
        The training targets, the thing you want to predict. In this project,
        this is another norm set which will be compared to the reference (X)
    verbose : bool
        Whether to display a progress bar or not. Defaults to False.
    break_after : int | None
        Break after this many pairwise comparisons. Useful for debugging.
        Defaults to None, which means all comparisons will be performed.
    normalize : bool
        Whether to normalize X_all, X_perm and y (normalize along each column).
        Defaults to True (you normally always want to do this).
    model : scikit-learn regression model | None
        The model to use. If None, the RidgeGCV model, given in ridge.py,
        will be used. You can also pass a scikit-learn pipeline
        here, which is useful if you want to add some preprocessing steps that
        should be done at each iteration of the leave-two-out loop.
    metric : str
        The distance metric to use. Any distance implemented in SciPy's
        spatial.distance module is supported. See the docstring of
        scipy.spatial.distance.pdict for the exhaustive list of possitble
        metrics. Here are some of the more useful ones:
        'euclidean' - Euclidean distance
        'sqeuclidean' - Squared euclidean distance (the default)
        'correlation' - Pearson correlation
        'cosine' - Cosine similarity
    cv_splits : generator of tuples | None
        By default, a leave-two-out cross-validation scheme is used. Use this
        parameter to specify some custom cross-validation scheme. The supplied
        generator should produce tuples of (train_set_index, test_set_index).
        When using this in combination with the X_perm parameter, make sure the
        number of iterations match.

    Returns
    -------
    pairwise_accuracies : 2D array of bool (n_stimuli, n_stimuli)
        Matrix containing the results for all pairwise comparisons.
        To tell whether, given 2 stimuli, the model could correctly tell which
        was which, you can perform a lookup in this matrix:
        pairwise_accuracies[word1_index, word2_index]
        (where word1_index < word2_index)
    model : a scikit-learn regression model
        The model trained on all samples. This is the same model as specied
        in the `model` input parameter. You can use this object to obtain the
        regression weights and such.
    target_scores : 1D array (n_targets,)
        For each target, a score between 0 and 1 of how well it could be
        predicted from the data.
    predicted_y : 2D array (n_items, n_ginter_features)
        The predicted targets. In this project, these are the Ginter feature
        vectors that are predicted from the brain data.
    patterns : 2D array (n_samples, n_features)
        Activity patterns that can be interpreted.

    Notes
    -----
     - Currently, each sample must belong to a unique label.
     - The zero-shot-learning approach works best when labels have multiple
       targets (i.e. y has multiple columns).

    See also
    --------
    The predictions_breakdown function can be used to compute accuracies given
    the pairwise_accuracies matrix produced by this function.
    """
    assert len(X) == len(y)
    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    assert X.shape[0] == y.shape[0] == n_samples

    if cv_splits is None:
        splits = list(LeavePOut(n_samples, 2))
    else:
        splits = list(cv_splits)
    n_iterations = len(splits)


    # Normalize the features and the targets
    if normalize:
        X = zscore(X, axis=0, ddof=1)
  
        y = zscore(y, axis=0, ddof=1)



    # Results will be saved to this pairwise matrix
    pairwise_accuracies = np.zeros((n_samples, n_samples), dtype=np.bool)

    # Prediction errors for each target are accumulated here
    prediction_errors = []

    # model to use
    if model is None:
        alphas = [.0000001, .000001, .00001, .0001, .001, .01, .1, .5, 1, 5,
                  10, 50, 100, 500, 1000, 10000, 20000, 50000, 100000, 500000,
                  1000000, 5000000, 10000000]
        model = RidgeGCV(alphas=alphas, alpha_per_target=True,
                         fit_intercept=True, normalize=False,
                         store_cv_values=True)

    if verbose:
        from progressbar import ProgressBar, Bar, SimpleProgress, ETA, Timer
        pb = ProgressBar(
            maxval=n_iterations,
            widgets=[Bar(), ' ', SimpleProgress(), ' ', Timer(), ' ', ETA()]
        ).start()

    # Store predictions
    predicted_y = np.zeros((n_samples, n_targets))

    i = 0
    for train_index, test_index in splits:

        X_iter = X

        model.fit(X_iter[train_index], y[train_index])
        prediction = model.predict(X_iter[test_index])
        accuracy = _leave_two_out_accuracy(y[test_index], prediction, metric=metric)

        # Store the result in the upper triangle of the pairwise matrix
        if test_index[0] < test_index[1]:
            pairwise_accuracies[test_index[0], test_index[1]] = accuracy
        else:
            pairwise_accuracies[test_index[1], test_index[0]] = accuracy

        # Store prediction
        predicted_y[test_index[0]] += prediction[0]
        predicted_y[test_index[1]] += prediction[1]

        # Used to calculate the "coefficient of determination" of the targets
        prediction_errors.append((prediction - y[test_index]) ** 2)

        if verbose:
            pb.update(i)

        i += 1

    if verbose:
        pb.finish()

    # Fit model to all data. This is the model that gets returned
    model.fit(X, y)

    # Compute the mean of the predicted targets (they have been computed twice
    # due to the leave-two-out scheme).
    predicted_y /= 2

    # Calculate the "coefficient of determination" (Eq. 4 in Sudre et al. 2012)
    # This value represents how well we are able to predict a target (from 0
    # to 1; 1 implying perfect prediction of the targets).
    target_scores = 1 - (np.mean(model.cv_values_.min(axis=2), axis=0) /
                         np.mean(y ** 2, axis=0))

    # Multivariate regression weights cannot be interpreted. Compute patterns
    # that *can* be.
    patterns = _patterns_from_weights(X, model.coef_, predicted_y, model.alpha_)

    return pairwise_accuracies, model, target_scores, predicted_y, patterns
