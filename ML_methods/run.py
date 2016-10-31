
# coding: utf-8

# # Final prediction (Only Code)
# ### Using 4 model ridge regression 

# In[1]:

# Useful starting lines
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from proj1_helpers import *
from ml_functions import *


# In[84]:

# builds the tX specific polynomial degree
def build_poly_d(tX_all_degree, degree):
    index = list()
    index.append(0)
    for i in range(0,29): # tX.shape[1] is 29, 
        for d in range(1,degree+1): # up to degree 10 for example
            index.append(i*10+d) # [1,11,...,281]
    return tX_all_degree[:,index]

# x is one column
def build_poly_one_column(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    size = (x.shape[0])*(degree)
    phi = list()
    
    for i in range(0, x.shape[0]):
        for j in range(1, degree+1):
            index = j + i*(degree+1)
            phi.append(np.power(x[i], j))
            
    return np.array(phi).reshape((1, size))

#polyX = np.full((tX.shape[0], tX.shape[1] * (degree+1)), 0.0)
# tX is a multidimensional array
def build_poly3(tX, degree):
    polyX = list()
    for row in tX:
        polyX.append(build_poly_one_column(row, degree))

    PolyXNP = np.array(polyX)
    PolyXNP2 = PolyXNP.reshape((PolyXNP.shape[0], PolyXNP.shape[2]))

    return PolyXNP2
    #Standardize
    PolyXNP2 = standardize(PolyXNP2)
    


# In[85]:

# standardize such that the mean is 0 and standard deviation is 1.
# X is a multi-dimensional array
def standardize2(X):
    #X = (X - np.mean(X)) / np.std(X)
    X = np.hstack((np.ones((X.shape[0],1)), X)) # adding column of ones for the first column
    return X


# In[106]:

from proj1_helpers import *
DATA_TRAIN_PATH = '../Data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


# In[87]:

#for col in tX.T:
    #med = np.median(col[col != -999])
    #col[col == -999] = med


# In[107]:

model_index = tX[:, 22] # model index is useful later on to select only the indices where categorical variable = 0,1,2,3

#remove model index column from tx
tX = np.delete(tX, 22, 1)


# In[108]:

# building a polynomial matrix of degree 10
tX_10 = build_poly3(tX,10)


# In[109]:

Fused = np.c_[y, tX_10]


# In[110]:

max_degree = 10
tX_list = list()
# takes a long time to run, run only once!!!
for degree in range(1,max_degree+1):
    tX_list.append(build_poly_d(Fused, degree))


# In[111]:

# degree 3
Fused_0 = tX_list[1][model_index == 0.0] # degree 2
y0 = Fused_0[:,0]
x0 = Fused_0[:,1:Fused_0.shape[1]]
x0 = standardize2(x0)

Fused_1 = tX_list[1][model_index == 1.0] # degree 2
y1 = Fused_1[:,0]
x1 = Fused_1[:,1:Fused_1.shape[1]]
x1 = standardize2(x1)

Fused_2 = tX_list[1][model_index == 2.0] # degree 2
y2 = Fused_2[:,0]
x2 = Fused_2[:,1:Fused_2.shape[1]]
x2 = standardize2(x2)

Fused_3 = tX_list[1][model_index == 3.0] # degree 2
y3 = Fused_3[:,0]
x3 = Fused_3[:,1:Fused_3.shape[1]]
x3 = standardize2(x3)


# In[113]:

x3.shape


# In[114]:

lambda0 = 0.01
lambda1 = 0.01
lambda2 = 0.01
lambda3 = 0.01

loss1, w1 = ridge_regression(y0, x0, lambda0)
loss2, w2 = ridge_regression(y1, x1, lambda1)
loss3, w3 = ridge_regression(y2, x2, lambda2)
loss4, w4 = ridge_regression(y3, x3, lambda3)

weights = (w1, w2, w3, w4)


# In[115]:

w1.shape


# In[116]:

DATA_TEST_PATH = '../Data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


# In[99]:

#for col in tX_test.T:
    #med = np.median(col[col != -999])
    #col[col == -999] = med


# In[117]:

#keep model index column seperate
model_index2 = tX_test[:, 22]

#remove model index column from tx
tX_test = np.delete(tX_test, 22, 1)


# In[118]:

tX_test_poly = build_poly3(tX_test, 2)
#Add a column of ones such that dim indexing match 


# In[119]:

tX_test_poly = standardize2(tX_test_poly)


# In[120]:

tX_test_poly.shape


# In[121]:

def predict_labels_ridge_4(weights, data, model_index):
    """Generates class predictions given weights, and a test data matrix"""
  
    y_pred = list()
    
    for i in range(0, model_index.shape[0]):
        if model_index[i] == 0.0:
            y_pred.append(data[i].dot(weights[0]))
        if model_index[i] == 1.0:
            y_pred.append(data[i].dot(weights[1]))
        if model_index[i] == 2.0:
            y_pred.append(data[i].dot(weights[2]))
        if model_index[i] == 3.0:
            y_pred.append(data[i].dot(weights[3]))
    
    y_final = np.array(y_pred)
                              
    y_final[np.where(y_final <= 0)] = -1
    y_final[np.where(y_final > 0)] = 1
    
    return y_final


# In[122]:

OUTPUT_PATH = '../Data/final_result2.csv' 
# TODO: fill in desired name of output file for submission
y_pred = predict_labels_ridge_4(weights, tX_test_poly, model_index2)
print(y_pred.shape)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


# In[ ]:



