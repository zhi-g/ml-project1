get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from proj1_helpers import *
from ml_functions import *

# Import the data 

DATA_TRAIN_PATH = '../Data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# model index is useful later on to select only the indices where categorical variable = 0,1,2,3
model_index = tX[:, 22] 

#remove model index column from tx
tX = np.delete(tX, 22, 1)

# building a polynomial matrix of degree 10
tX_10 = build_poly3(tX,10)

# fuse tX and y to keep rows assignment when building 4 model
Fused = np.c_[y, tX_10]

max_degree = 10
tX_list = list()

for degree in range(1,max_degree+1):
    tX_list.append(build_poly_d(Fused, degree))

#Build the 4 ys and x0 for each model and add a column of 1s
#(standardize2 doesn't actually standardize for this result)
    
#We chose only degree polynomial of degree up to 2 => list[1]
Fused_0 = tX_list[1][model_index == 0.0] 
y0 = Fused_0[:,0]
x0 = Fused_0[:,1:Fused_0.shape[1]]
x0 = standardize2(x0)

Fused_1 = tX_list[1][model_index == 1.0] 
y1 = Fused_1[:,0]
x1 = Fused_1[:,1:Fused_1.shape[1]]
x1 = standardize2(x1)

Fused_2 = tX_list[1][model_index == 2.0]
y2 = Fused_2[:,0]
x2 = Fused_2[:,1:Fused_2.shape[1]]
x2 = standardize2(x2)

Fused_3 = tX_list[1][model_index == 3.0]
y3 = Fused_3[:,0]
x3 = Fused_3[:,1:Fused_3.shape[1]]
x3 = standardize2(x3)

#Run ridge regression on 4 model to get a list of all the weight
lambda0 = 0.01
lambda1 = 0.01
lambda2 = 0.01
lambda3 = 0.01

loss1, w1 = ridge_regression(y0, x0, lambda0)
loss2, w2 = ridge_regression(y1, x1, lambda1)
loss3, w3 = ridge_regression(y2, x2, lambda2)
loss4, w4 = ridge_regression(y3, x3, lambda3)

weights = (w1, w2, w3, w4)

#Import the test set
DATA_TEST_PATH = '../Data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#keep model index column seperate
model_index2 = tX_test[:, 22]

#remove model index column from tx
tX_test = np.delete(tX_test, 22, 1)

#Build the polynomial of degree 2 which gave us best results for test set
tX_test_poly = build_poly3(tX_test, 2)

#Add a column of ones such that dimension matches, it's not standardizing in this implementation for this result
tX_test_poly = standardize2(tX_test_poly)

OUTPUT_PATH = '../Data/final_result2.csv' 
#Predict the ouput given the test data
y_pred = predict_labels_ridge_4(weights, tX_test_poly, model_index2)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)




