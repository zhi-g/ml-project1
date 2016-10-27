# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)

    
def compute_cost_ll(y, tx, w):
    """Computes cost as negative value of the log-likelihood."""
    tXW = np.dot(tx, w)
    return np.dot(-y.T, tXW) + np.mean(np.log(1 + np.exp(tXW)))
