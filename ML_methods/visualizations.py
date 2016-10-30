from ml_functions import *
from costs import *
import numpy as np
from plots import *
from helpers import *

def cross_validation_demo():
    """Generating the cv visualization for one example."""
    x, y = load_data()
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    # cross validation: TODO
    for lambda_ in lambdas:
        loss_tr, loss_te = cross_validation_rr(y, x, k_indices, k_fold, lambda_, degree)
        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)
    
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)


def bias_variance_demo(): 
    """Usage: from helpers import bias_variance_demo, and bias_variance_demo()
       Returns a sample bias_variance plot of a specific sine function
       need to edit this for specific x,y"""
    """Generating the visualization for one example."""
    # define parameters
    seeds = range(10)
    num_data = 1000
    ratio_train = 0.05
    degrees = range(1, 10) # 1 to 9
    
    # define list to store the variable
    rmse_tr = np.empty((len(seeds), len(degrees))) #train... #what's np.empty??
    rmse_te = np.empty((len(seeds), len(degrees))) #test
    
    # generating the data
    for index_seed, seed in enumerate(seeds): # for one exact point.., 100 different seeds
        np.random.seed(seed)
        x = np.linspace(0.1, 2 * np.pi, num_data) # x is total amount of data
        y = np.sin(x) + 0.3 * np.random.randn(num_data).T # f(x) + 2 is also fixed
        
        # split between test and train
        train, test = split_data(x, y, ratio_train)
        y_train, x_train = train[:,0:1],train[:,1:10]
        y_test, x_test = test[:,0:1],test[:,1:10]

        #for degree 1 to 9
        for index_degree, degree in enumerate(degrees):
            # get the weights using 
            tX_train = build_poly(x_train, degree)
            tX_test = build_poly(x_test, degree)
            weights = least_squares(y_train, tX_train) 
            
            # compute train and test RMSE
            rmse_tr[index_seed,index_degree] = np.sqrt(2*compute_loss(y_train,tX_train,weights))
            rmse_te[index_seed,index_degree] = np.sqrt(2*compute_loss(y_test,tX_test,weights))
        

    bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te)