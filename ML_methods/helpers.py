# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x

def standardize_filtered(tx):
    """Standardize the original data set with -999 values set to zero."""
    N = np.shape(tx)[1];
    for i in range(N): 
        f = tx[:,i] == -999
        d = tx[:,i] != -999
        temp = tx[d, i]
        mean = np.mean(temp)
        std = np.std(temp)
        tx[d,i] = [(x - mean) / std for x in tx[d,i]] 
        tx[f,i] = [0 for _ in tx[f,i]]

    return tx

def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_indices, k, classification_func, err_func):
    """Estimates the error of a given classification function. """
    x_test = tx[k_indices[k],:]
    y_test = y[k_indices[k]]

    x_train = np.delete(tx, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k])

    train_loss, w = classification_func(y_train, x_train)
    test_loss = err_func(y_test, x_test, w)

    return train_loss, test_loss


def k_fold(y, tx, k, classification_func, err_func):
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for k in range(len(k_indices)):
        tr_loss, te_loss = cross_validation(y, tx, k_indices, k, classification_func, err_func)
        rmse_tr.append(tr_loss)
        rmse_te.append(te_loss)

    return np.mean(rmse_tr), np.mean(rmse_te)
