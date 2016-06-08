__author__ = 'how'

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle
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
    df.to_csv('func_change_cor_2_new.csv')


def sin_scale(x):
    return np.sin(5*x)

def log_scale(x):
    return np.log2(1+0.5*x)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## note: The functions below are designed for when x is a numpy array, and may not work for scalar x
def sineFunctionGenerator(phase=0, period=1, scale=1):
    def f(x):
        return scale * np.sin( (x - phase) * 2 * np.pi / period)
    return f

def linFnGenerator(slope=1, startVal=0, xmin=0, xmax=1):

    def f(x):
        ret = slope * (x - xmin) + startVal
        ret[np.logical_or(x<xmin, x>xmax)] = 0
        return ret
    return f

def stepFnGenerator(stepLoc=0):
    def f(x):
        ret = np.zeros(len(x))
        ret[x >= stepLoc] = 1
        return ret
    return f

def linearCombGenerator(fns, coefs):
    def f(x):
        return 2*sum(fns[i](x) * coefs[i] for i in range(len(fns)))
    return f

def VarlinearCombGenerator(fns, coefs):
    def f(x):
        return (np.abs(sum(fns[i](x) * coefs[i] for i in range(len(fns))))+0.1)/2
    return f

## Define a master set of basis functions
def generate_random():
    allBasisFns = [sineFunctionGenerator(phase=a, period=b) for a in np.linspace(0,.4,10) for b in np.linspace(.1,.4,10)]
    allBasisFns += [linFnGenerator(slope=s, xmin=x, xmax=x+.5) for s in np.linspace(-4,4,10) for x in np.linspace(0,1,10)]
    allBasisFns += [stepFnGenerator(stepLoc=s) for s in np.linspace(0,1,100)]

    ## Create a random target function by selecting some basis functions and choosing random coefficients
    seed = np.random.RandomState(2)
    seed.shuffle(allBasisFns)
    numBasisToKeep = 4
    basisFns = allBasisFns[1:numBasisToKeep+1]
    print basisFns
    coefs = seed.randn(numBasisToKeep)
    targetFn = linearCombGenerator(basisFns, coefs)

    ## Plot the target function on [0,1]
    x = np.arange(0.0, 1.0, 0.01)
    y = targetFn(x)
    #plt.plot(x, y)
    #plt.show()

    #Plot the target function on [0,1]

    seed.shuffle(allBasisFns)
    numBasisToKeep = 4
    basisFns = allBasisFns[1:numBasisToKeep+1]
    coefs = seed.randn(numBasisToKeep)
    varFn = VarlinearCombGenerator(basisFns, coefs)
    var = varFn(x)
    #plt.plot(x, var)
    #plt.show()



    return targetFn, varFn

def generate_4(low = 0., high=1., N=800):
    targetFn, varFn = generate_random()
    seed = np.random.RandomState(12542378)
    x = seed.uniform(low,high,N)
    y = [seed.normal(loc=targetFn([i])[0], scale=varFn([i])[0]) for i in x]
    #plt.scatter(x, y)
    #plt.show()

    df = pd.DataFrame(data={'x': x, 'y': y})
    df.to_csv('func_change_cor_3.csv')

def viz_par(mean_func, var_func):
    #plot mu(x) and var(x) as x increase
    x = np.array(range(500))/500.
    mean_x = mean_func(x)
    var_x = var_func(x)
    plt.plot(x, mean_x, 'b')
    plt.plot(x, var_x, 'r')
    plt.xlabel('x')
    plt.ylabel('parameters')
    plt.legend(['mean(x)', 'std_err(x)'])
    plt.title('Mean/Variance function value of x')
    plt.show()

def main():
    import cPickle
    #generate_3(400, lambda x: np.sin(2.5*x*3.14)*np.sin(1.5*x*3.14), lambda x: 0.01+0.25*(1-np.sin(2.5*x*3.14))**2)
    #generate_3(400, sin_scale, log_scale)
    generate_4(N=400)

    #viz_par(lambda x: np.sin(2.5*x*3.14)*np.sin(1.5*x*3.14), lambda x: 0.01+0.25*(1-np.sin(2.5*x*3.14))**2)
    print "Hello World"

    targetFn, varFn = generate_random()
    viz_par(targetFn, varFn)




