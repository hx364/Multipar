__author__ = 'how'

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
"""
To generate data for test
x ~ uniform(0,1)
y ~ R
For regression setting
"""


def generate_1(N, low=0., high=1., scale=0.2):
    """
    Fixed covariance, linear in x
    x: uniform[0,1] y|x ~ N(x, theta)
    :param variance: scale of noise
                N  : number of data
    :return:
    """
    x = np.random.uniform(low,high,N)
    y = [np.random.normal(loc=i, scale=scale) for i in x]

    plt.scatter(x, y)
    plt.show()
    plt.close()

    df = pd.DataFrame(data={'x': x, 'y': y})
    df.to_csv('linear_fix_cor.csv')


def generate_2(N, f, low=0., high=1., scale=0.2):
    x = np.random.uniform(low,high,N)
    y = [np.random.normal(loc=f(i), scale=scale) for i in x]

    plt.scatter(x, y)
    plt.show()
    plt.close()

    df = pd.DataFrame(data={'x': x, 'y': y})
    df.to_csv('func_fix_cor.csv')


def generate_3(N, f, g, low=0., high=1.):
    """
    generate data with complicated mean and changing covariance
    :param N: Number of data points
    :param f: mean function of noise
    :param g: variance function of noise
    :param low:
    :param high:
    :return:
    """
    x = np.random.uniform(low,high,N)
    y = [np.random.normal(loc=f(i), scale=g(i)) for i in x]

    plt.scatter(x, y)
    plt.show()
    plt.close()

    df = pd.DataFrame(data={'x': x, 'y': y})
    df.to_csv('func_change_cor.csv')


def sin_scale(x):
    return np.sin(5*x)

def log_scale(x):
    return np.log2(1+0.5*x)
"""
def main():
    generate_3(200, sin_scale, log_scale)

main()
"""