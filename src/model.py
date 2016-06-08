__author__ = 'how'
from utils import *
import climate
import theanets
import theano.tensor as TT
import theano

def viz_clf(x_train, x_test, y_train, y_test, clf = None):
    """
    visualize the data, with classifier embedded
    """
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    color = ['r']*len(x_train)+['b']*len(y_train)
    plt.scatter(x, y, c=color, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data fitted with fixed variance')
    plt.savefig('fixed_var.jpg')

    #visualize the model in plot
    if clf:
        x_c = np.array(range(500))/500.
        y_c = clf.predict(expand(x_c))
        plt.plot(x_c, y_c)

    #print the nll value
    nll = np.sum(compute_neglog(x_test[i], y_test[i]) for i in range(len(y_test)))
    print "NNL on test data: %r" %(nll)
    plt.savefig('fixed_var.jpg')
    plt.show()


def linear_regression(x, y, c=0.1):
    from sklearn.linear_model import Ridge
    clf = Ridge(alpha=c)
    clf.fit(x,y)
    return clf






def main():
    x_train, x_test, y_train, y_test = read_data('func_change_cor.csv')
    print np.shape(expand(x_train))

    """
    for i in [0, 0.00001,0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        clf = linear_regression(expand(x_train), y_train, c=i)
        pred = clf.predict(expand(x_test))
        nll = np.sum(compute_neglog(x_test[i], pred[i]) for i in range(len(pred)))
        print "C: %r   NNL on test data: %r" %(i, nll)
    """


    #clf = linear_regression(expand(x_train), y_train, c=0.001)
    #pred = clf.predict(expand(x_test))
    #viz_clf(x_train, x_test, y_train, y_test, clf=clf)]
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'





main()

