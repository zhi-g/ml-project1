# -*- coding: utf-8 -*-
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
def least_squares_GD(y, tx, initial_w, gamma, max_iters): 
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

def build_poly2(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    size = (x.shape[0])*(degree)
    #phi = np.full((1, size), 0.0)
    phi = list()
    
    for i in range(0, x.shape[0]):
        for j in range(1, degree+1):
            index = j + i*(degree+1)
            #np.put(phi, index, np.power(x[i], j))
            phi.append(np.power(x[i], j))
            
    return np.array(phi).reshape((1, size))

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

def calculate_loss_negative_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    #loss = np.log(1 + np.exp(tx.dot(w))) - y * tx.dot(w)
    #return np.sum(loss, axis=0)
    return 0

def sigmoPred(t):
    """apply sigmoid function on t."""
    if t >= 50:
        return 1
    if t <= -50:
        return 0
    else:
        return 1 / ( 1 + np.exp(t))

    
def sigmoid(x):
    """Implement sigmoid function for ridge regression."""
    for i in x:
        for j in i:
            j = sigmoPred(j)
    return x

def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    y = y.reshape((y.shape[0], 1))
    loss = calculate_loss_negative_log_likelihood(y, tx, w)
    
    sig = sigmoid(np.dot(tx, w))
    sigy = sig - y
    
    ret = tx.T.dot(sigy)
    
    return loss, ret
    #return loss, tx.T.dot(sigmoid(tx.dot(w) - y))


def logistic_regression_test(y, tx, gamma, max_iter):

    w = np.zeros((tx.shape[1], 1))
    # start the logistic regression
    for iter in range(max_iter):
        #loss, w = learning_by_newton_method(y, tx, w, gamma)[0]
        loss, grad = calculate_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        
    print(loss)
    return loss, w
    
    
def conv_logistic_regression_test(y, tx, gamma):
    
    w = np.zeros((tx.shape[1], 1))
    # start the logistic regression
    last_loss = np.inf
    while True:
        #loss, w = learning_by_newton_method(y, tx, w, gamma)[0]
        loss, grad = calculate_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        if(loss <= 0.0):
            break
            
        last_loss = loss
            
    print(loss)
    return loss, w
    
def pen_logistic_regression_test(y, tx, lamb, gamma, max_iter):
    # init parameters
    w = np.zeros((tx.shape[1], 1))
    N = tx.shape[0]

    print(tx.shape)
    print(y.shape)
    print(w.shape)
    
    # start the logistic regression
    for iter in range(max_iter):
        loss, grad = calculate_gradient_logistic(y, tx, w)
        
        #We don't want to penalize w0 ?sss
        m2grad = grad + 2* lamb * w
        #m2grad[0] = grad[0]
        w = w - gamma * (m2grad)
       
        
    print(loss)
    return loss, w

'''
def sigmoid2(x):
    """Implement sigmoid function for ridge regression."""
    elem = list()
    for i in range(0, x.shape[0]):
        if x[i] >= 15:
            elem.append(1)
        elif x[i] < -15:
            elem.append(0)
        else:
            elem.append(1 / ( 1 + np.exp(x[i])))

    array = np.array(elem).reshape((x.shape[0], ))
    return array

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    N = tx.shape[0]
    S = (sigmoTest(tx.dot(w)) * (1-sigmoTest(tx.dot(w)))).reshape((N, 1)) * tx
    return tx.T.dot(S)

def learning_by_pen_newton_method(y, tx, w, lamb, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, grad = calculate_gradient_logistic(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    
    pen_grad = grad + 2.0 * lamb * w
    pen_hessian = hessian + 2.0 * lamb
    
    hgrad = np.linalg.inv(pen_hessian).dot(pen_grad)
    
    return loss, w - gamma * hgrad
    
   
'''
          