# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from ml_functions import *

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    print(x.max())
    
    #tx = np.hstack((np.ones((x.shape[0],1)), x))
    return x#, mean_x, std_x

def standard(x):
    x = (x - x.min(0)) / x.ptp(0)
    return x

def standard2(X):
    
    X = (X - np.mean(X)) / np.std(X)
    return X

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

#return both sets as two different variable: train_data, test_data
def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    data = np.c_[y, x]
    
    np.random.shuffle(data)
    
    total_rows = x.shape[0]
    train_rows = int(np.floor(total_rows * ratio))
    test_rows = total_rows - train_rows
    
    #This should be the case, check that I didn't make a mistake above
    assert(total_rows == train_rows + test_rows)

    train_data = data[:train_rows]
    test_data = data[train_rows:(train_rows + test_rows)]

    return train_data, test_data
            
            
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


def cross_validation_rr(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    # empty array for computing the mean for each hold
    loss_tr_arr = [] 
    loss_te_arr = [] 
    
    for k in range(0,k):
        # get k'th subgroup in test, others in train
        x_test = x[k_indices[k]]
        y_test = y[k_indices[k]]
        x_train = np.delete(x, k_indices[k], axis=0)
        y_train = np.delete(y, k_indices[k])

        # form data with polynomial degree
        tx_train = build_poly(x_train,degree)
        tx_test = build_poly(x_test,degree)

        # ridge regression
        loss_tr, weights = ridge_regression(y_train, tx_train, lambda_)
        loss_te = compute_loss(y_test,tx_test,weights)
        loss_tr_arr.append(loss_tr)
        loss_te_arr.append(loss_te)
        
    # calculate the loss for train and test data
    loss_tr = np.mean(loss_tr_arr)
    loss_te = np.mean(loss_te_arr)
    
    return loss_tr, loss_te


def load_data(): # for visualization in cross_validation_demo() 
    """load data."""
    data = np.loadtxt("dataEx3.csv", delimiter=",", skiprows=1, unpack=True)
    x = data[0]
    y = data[1]
    return x, y