# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from helpers import *
from ml_functions import build_poly2, sigmoPred, sigmoid


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

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


def predict_labels_logistic_4_model(weights, data, degree, model_index):
    """Generates class predictions given weights, and a test data matrix"""
  
    y_pred = np.array([])
    
    #These appends take too much time TODO use list and then transform to np.array if time
    for i in range(0, model_index.shape[0]):
        
        if model_index[i] == 0.0:
            y_pred = np.append(y_pred, sigmoPred(data[i].T.dot(weights[0])))
        if model_index[i] == 1.0:
            y_pred = np.append(y_pred, sigmoPred(data[i].T.dot(weights[1])))
        if model_index[i] == 2.0:
            y_pred = np.append(y_pred, sigmoPred(data[i].T.dot(weights[2])))
        if model_index[i] == 3.0:
            y_pred = np.append(y_pred, sigmoPred(data[i].T.dot(weights[3])))
    
    print(y_pred)
    
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def predict_labels_logistic_single(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(np.dot(data, weights))
    print(y_pred)
    
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred



def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

