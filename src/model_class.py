__author__ = 'how'

#define the three model classes for density estimation
#GLM, NN, GBM


__author__ = 'how'
import numpy as np
from utils import read_data, expand, compute_neglog
import numpy as np
import matplotlib.pyplot as plt
from data_generate import log_scale

#############################################
class linear_model():
    def __init__(self, d_start=0, d_end=3):
        self.d_start = d_start
        self.d_end = d_end



    def fit(self, X, y):

    def predict(self, x):

    def score(self, X, y):
        #Return the NLL score
        f_1 = np.dot(X, P)
        f_2 = np.exp(-(np.dot(X, P)))*((y-np.dot(X, W))**2)
        return sum(f_1+f_2)/2.





def evaluate_nl(X, y, P, W):
    """
    compute the loss function value
    f = 1/2*sum(P.T*X_i)+1/2*sum((y_i-W.T*X_i)**2*exp(-P.T*X_I))
    X: array of (num_instances, num_feat)
    y: array of (num_instances)
    P: array of (num_feat)
    W: array of (num_feat)
    """
    #print X.shape, y.shape
    f_1 = np.dot(X, P)
    f_2 = np.exp(-(np.dot(X, P)))*((y-np.dot(X, W))**2)
    return sum(f_1+f_2)/2.

def optimize_par(X, y, X_val, y_val, tol=0.0001):

    #get stats and initialize P
    num_instances, num_feat = X.shape
    W = np.ones(num_feat)
    P = np.ones(num_feat)
    X_sum = np.sum(X, axis=0)

    print evaluate_nl(X, y, P, W)

    for i in range(5000):
        #print "Iter %r" %i
        #optimize on W
        old_loss_func = evaluate_nl(X, y, P, W)
        mu_mat = np.exp(-np.dot(X,P))
        y_pseudo = y*np.sqrt(mu_mat)
        #X_pseudo = X*np.sqrt(np.array([mu_mat,mu_mat,mu_mat, mu_mat]).T)
        X_pseudo = X*np.sqrt(np.repeat(np.expand_dims(mu_mat, axis=1), num_feat, axis=1))

        W = np.dot(np.dot(np.linalg.inv(np.dot(X_pseudo.T, X_pseudo)), X_pseudo.T), y_pseudo)
        #print W
        #print evaluate_nl(X, y, P, W)

        #optimize on P
        m_pseudo = np.log((y - np.dot(X, W))**2)
        #P_init = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), m_pseudo)
        P = optimize_on_P(X, P, m_pseudo, alpha=0.01)
        print "Epoch: %r Train Cost: %.3f, Val cost: %.3f" %(i, evaluate_nl(X, y, P, W), evaluate_nl(X_val, y_val, P, W))
        error = old_loss_func - evaluate_nl(X, y, P, W)


        if error < tol:
            return W, P

    return W, P


def optimize_on_P(X, P, m_pseudo, alpha=0.01, tol=0.001):
    """
    Use gradient Descent to find the optimal P
    """
    X_sum = np.sum(X, axis=0)
    #print "func_P: %r" %func_P(X, P, m_pseudo)
    error=100
    for i in range(200):
        old_func_val = func_P(X, P, m_pseudo)
        grad_P = X_sum - np.dot(X.T, np.exp(m_pseudo-np.dot(X, P)))
        P = P - grad_P*alpha

        if func_P(X, P+grad_P*alpha, m_pseudo) -  func_P(X, P, m_pseudo) <tol:
            #print "func_P: %r" %func_P(X, P, m_pseudo)
            return P

    #print "func_P: %r" %func_P(X, P, m_pseudo)
    return P




def func_P(X, P, m_pseudo):
    """
    Evaluate the loss function value given X, P
    """
    f_val = np.sum(np.dot(X, P) + np.exp(m_pseudo-np.dot(X, P)))
    return f_val


def viz_clf(x_train, x_test, y_train, y_test, W, P):
    """
    visualize the data, with classifier embedded
    """
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    color = ['r']*len(x_train)+['b']*len(y_train)
    plt.scatter(x, y, c=color, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data fitted with Changing parametric variance')

    #visualize the model in plot
    #plot mean line
    x_c = np.array(range(500))/500.
    y_c = np.dot(expand(x_c, 0, 3), W)

    #plot variance line
    variance_c = np.exp(np.dot(expand(x_c, 0, 3), P))

    plt.plot(x_c, y_c, 'b')
    plt.plot(x_c, y_c+np.sqrt(variance_c),'r')
    plt.plot(x_c, y_c-np.sqrt(variance_c), 'r')

    #print the nll value
    mean_test = np.dot(expand(x_test, 0, 3), W)
    var_test = np.exp(np.dot(expand(x_test, 0, 3), P))
    nll = np.sum(compute_neglog(y_test[i], mean_test[i], var_test[i]) for i in range(len(y_test)))

    true_mean = np.sin(5*x_test)
    true_var = np.log2(1+0.5*x)
    true_nll = np.sum(compute_neglog(y_test[i], true_mean[i], true_var[i]) for i in range(len(y_test)))
    print "NNL on test data: %r" %(nll)
    print "Optimal NNL %r" %(true_nll)

    plt.show()

def plot_var(P):
    x_c = np.array(range(500))/500.
    y_c = [log_scale(i) for i in x_c]
    variance_c = np.exp(np.dot(expand(x_c, 0, 3), P))
    p1, = plt.plot(x_c, y_c, 'b')
    p2, = plt.plot(x_c, variance_c, 'r')
    plt.legend([p1, p2], ['True Variance', 'Estimated Variance'])
    plt.show()



def main():
    x_train, x_test, y_train, y_test = read_data('func_change_cor.csv')

    W, P = optimize_par(expand(x_train, 0, 3), y_train, expand(x_test, 0, 3), y_test) # W.T*X is mean function, exp(P.t*X) is scale function
    #print W
    viz_clf(x_train, x_test, y_train, y_test, W, P)
    plot_var(P)

main()


