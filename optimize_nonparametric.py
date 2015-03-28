__author__ = 'how'
import numpy as np
from utils import *
import pandas as pd
import numpy as np

##########
#This script is to optimize the linear regression with nonparametric changing variance
#The variance is estimated by a moving average of sample variance

def estimate_variance(X, y, W, n):
    """
    estimate the variance of each point by the moving average of sample variance
    given X, y, W
    variance = mean((y_i - W.T*x_i)**2)
    :return: variance, of shape(num_instances)
    """
    #Sort X and y, from min to max
    order = np.argsort(X[:,1])
    X, y = X[order], y[order]
    variance = pd.Series((y - np.dot(X, W))**2)
    variance = pd.stats.moments.rolling_mean(variance, window=n, min_periods=1, center=True)
    variance = np.asarray(variance)
    return variance[np.argsort(order)]


def evaluate_nl(X, y, W, var):
    """
    Compute the loss function value
    X: (num_instance, num_feat)
    y: (num_instance)
    W: (num_feat)
    var: (num_instance)
    """
    return np.sum(np.log(var)+ (y - np.dot(X, W))**2/var)

def optimize_nonpar(X, y, tol=0.0001):

    #get stats and initialize P
    num_instances, num_feat = X.shape
    W = np.ones(num_feat)
    var = np.ones(num_instances)
    X_sum = np.sum(X, axis=0)

    print evaluate_nl(X, y, W, var)

    for i in range(2):

        #optimize on W
        var_mat = np.repeat(np.expand_dims(var, axis=1), num_feat, axis=1)
        X_pseudo= X/np.sqrt(var_mat)
        y_pseudo = y/np.sqrt(var)

        W = np.dot(np.dot(np.linalg.inv(np.dot(X_pseudo.T, X_pseudo)), X_pseudo.T), y_pseudo)
        print W
        print evaluate_nl(X, y, W, var)

        #optimize on var
        var = estimate_variance(X, y, W, 5)
        print var
        print evaluate_nl(X, y, W, var)



def main():
    x_train, x_test, y_train, y_test = read_data('func_change_cor.csv')

    W, P = optimize_nonpar(expand(x_train), y_train)



main()



