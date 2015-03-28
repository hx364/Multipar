__author__ = 'how'

#define helper functions
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def read_data(file_name):
    """
    read the data file, split into train, and test set
    """
    df = pd.read_csv(file_name)
    x = np.array(df.x)
    y = np.array(df.y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state=36)
    return x_train, x_test, y_train, y_test


def expand(x, d_start, d_end):
    #expand the data to ,d-polynomial
    #start at x**d_start, end at x**d_end
    x_list = [x**i for i in range(d_start, d_end+1)]
    x = np.asarray(x_list).T
    return x

def compute_neglog(y, loc, scale):
    """
    Compute the neglog of point(x,y)
    """
    prob = norm(loc, scale).pdf(y)
    prob = np.min([-np.log(prob),10])
    prob -= np.log(2*np.pi)/2
    return prob

def NLL_loss(X, y):
    return np.sum([compute_neglog(X[i], y[i]) for i in range(len(y))])