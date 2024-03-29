# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np
from costs import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - (np.dot(tx,w))
    N = tx.shape[0]
    return (-1 / N) * np.dot(tx.T, e) 

# TODO check if we have to comply with the project description methods which would mean we have to
# get rid of the initial_w param
def least_squares_GD(y, tx, initial_w, gamma, max_iters): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    if initial_w == None:
        ws = np.zeros(np.zeros(30))
    else:
        ws = [initial_w]
    
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        loss = compute_loss(y, tx, w)
        
        w = w - gamma*compute_gradient(y, tx, w)

        ws.append(np.copy(w))
        losses.append(loss)
        
    return losses, ws

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    a = (np.dot(tx, w))
    e = y - a
    N = len(y)
    return (-1 / N) * np.dot(tx.T, e) 

# TODO check if we have to comply with the project description methods which would mean we have to
# get rid of the initial_w and batch_size param
def least_squares_SGD(y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        
        loss = compute_cost(y, tx, w)
        w = w - gamma * gradient
        ws.append(np.copy(w))
        losses.append(loss)
        
    return losses, ws

def least_squares(y, tx):
    """calculate the least squares solution."""
    if (tx.ndim == 1):
        tx = tx.reshape((tx.shape[0], 1))
    
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    
    # we use solve becaue linalg.inv gives numerical error with big numbers
    return np.linalg.solve(a,b)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.array([])
    for i in range(0, x.shape[0]):
        for j in range(0, degree+1):
            phi = np.append(phi, np.array([x[i] ** j]))

    phi = phi.reshape([x.shape[0], degree+1])
    return phi

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


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # Here we assume tx polynomial usually
    
    # Checking that if tx is of shape (#row,) we make it (#row, 1) for mat muliplication
    N = tx.shape[0]
    if tx.ndim == 1:
        tx = tx.reshape((N,1))
    M = tx.shape[1]
    
    id_mat = np.identity(M)
    #Because we don't want to penalize w0 (Asked TA) TODO: check
    id_mat[0][0] = 0
    
    x_inv = np.dot(tx.T, tx)
    id_mult = (lamb * (2 * N )) * id_mat #if I copy formula from slides
    
    #Solve again to compute matrix inverses
    a = np.linalg.solve(x_inv + id_mult, np.dot(tx.T, y))
    
    return a    


def sigma(x):
    """Implement sigmoid function for logistic regression."""
    z = np.exp(x)
    return  z / (1 + z)


def logistic_regression(y, tx, gamma, max_iters):
    "Implement ridge regression with gradient descent."
    N = np.shape(tx)[0]
    if tx.ndim == 1:
        tx = tx.reshape((N,1))
    M = tx.shape[1]
    w = np.zeros(M)
    for i in range(max_iters):
        gradient = (-1/N) * tx.T.dot(y - sigma(tx.dot(w)))
        w = w - gamma*gradient

    return w[1:]

def reg_logistic_regression(y, tx, lamb, gamma, max_iters):
    """Implement regularized logistic regression."""
    data_size = len(y)

    w = np.zeros(np.shape(tx)[1])
    w_star = w
    min_err = -1

    for i in range(max_iters):
        wPrim = w
        w[0] = 0

        err = compute_cost_ll(y, tx, w) + lamb * np.dot(w.T, w)
        grad = np.dot(-tx.T, y - sigma(np.dot(tx, w))) / data_size

        w = w - gamma * grad
        if (err < min_err or min_err == -1):
            min_err = err
            w_star = w

    return err, w_star


