
# coding: utf-8

# In[ ]:


"""a function used to compute the loss."""

import numpy as np
from costs import compute_loss, compute_cost_ll

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - (np.dot(tx,w))
    N = tx.shape[0]
    return (-1 / N) * np.dot(tx.T, e) 

# TODO check if we have to comply with the project description methods which would mean we have to
# get rid of the initial_w param
def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        loss = compute_loss(y, tx, w)
        
        w = w - gamma * compute_gradient(y, tx, w)

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
def least_squares_SGD(y, tx, initial_w, max_iter, gamma):
    """Stochastic gradient descent algorithm."""
    batch_size = 10
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iter):
        
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


def ridge_regression(y, tx, lambda_): # edited this function to return both loss and weights
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
    id_mult = (lambda_ * (2 * N )) * id_mat #if I copy formula from slides
    
    #Solve again to compute matrix inverses
    a = np.linalg.solve(x_inv + id_mult, np.dot(tx.T, y))
    
    return compute_loss(y, tx, a), a 

def logistic_regression(y, tx, initial_w, max_iter, gamma ):

    threshold = 1e-8
    losses = []
    w = initial_w 
    
    # start the logistic regression
    for iter in range(max_iter):
        #loss, w = learning_by_newton_method(y, tx, w, gamma)[0]
        loss, grad = calculate_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return losses, w
    
    
def reg_logistic_regression(y, tx, lamb, initial_w, max_iter, gamma):
    # init parameters
    w =  initial_w 
    threshold = 1e-8
    N = tx.shape[0]
    losses = []
    
    # start the logistic regression
    for iter in range(max_iter):
        loss, grad = calculate_gradient_logistic(y, tx, w)
        
        #We don't want to penalize w0 ?sss
        m2grad = grad + 2* lamb * w
        #m2grad[0] = grad[0]
        w = w - gamma * (m2grad)
        
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
       
    return losses, w

