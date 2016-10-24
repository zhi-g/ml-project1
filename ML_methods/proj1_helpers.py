# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from ml_functions import build_poly2


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


def predict_labels3(weights, data, degree):
    """Generates class predictions given weights, and a test data matrix"""
  
    model_id = (degree+1)*23 - 2
    y_pred = np.array([])
    
    #Use list to calculate polynomial faster
    polyX = list()
    for row in data:
        polyX.append(build_poly2(row, degree))

    PolyXNP = np.array(polyX)
    PolyXNP2 = PolyXNP.reshape((PolyXNP.shape[0], PolyXNP.shape[2]))
    
    
    #These appends take too much time TODO use list and then transform to np.array
    for row in PolyXNP2:
        if row[model_id] == 0.0:
            y_pred = np.append(y_pred, np.dot(row, weights[0]))
        if row[model_id] == 1.0:
            y_pred = np.append(y_pred, np.dot(row, weights[1]))
        if row[model_id] == 2.0:
            y_pred = np.append(y_pred, np.dot(row, weights[2]))
        if row[model_id] == 3.0:
            y_pred = np.append(y_pred, np.dot(row, weights[3]))
       
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def predict_labels2(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    
    y_pred = np.array([])
    
    for row in data:
        if row[22] == 0.0:
            y_pred = np.append(y_pred, np.dot(row, weights[0]))
        if row[22] == 1.0:
            y_pred = np.append(y_pred, np.dot(row, weights[1]))
        if row[22] == 2.0:
            y_pred = np.append(y_pred, np.dot(row, weights[2]))
        if row[22] == 3.0:
            y_pred = np.append(y_pred, np.dot(row, weights[3]))
       
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
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

