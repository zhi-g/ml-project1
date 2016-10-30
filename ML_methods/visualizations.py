from ml_functions import *
from costs import *
import numpy as np
from plots import bias_variance_decomposition_visualization

# Usage: from visualizations import bias_variance_demo, and bias_variance_demo
# Returns a sample bias_variance plot of a specific sine function
# need to edit this for specific x,y

def bias_variance_demo(): 
    """The entry."""
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