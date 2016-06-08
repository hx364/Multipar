from utils import read_data, compute_neglog, expand
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from data_generate import sin_scale, log_scale, generate_random

#There are two functional parameters
#O(x) = log(theta**2) for variance estimation
#u(x) for mean estimation
#g(x) is the gradient of l(x) respect to O(x)
#h(x) is the gradient of l(x) respect to u(x)


class U():
    #A class for the mean estimation for GBM
    def __init__(self, x, tree_max_depth=5):
        #x is a numpy array, the input of u
        self.x = x
        self.tree_max_depth = tree_max_depth
        self.gradient = np.zeros(x.shape[0])
        self.output = np.zeros(x.shape[0])
        self.model = []
        self.shrinkage = 0.1

    def fit(self, x, y, var, model = None):
        #based on g(x), find one base estimator to -gm
        #now use a base tree learner

        #Create a new trees

        if model == None:
            base = DecisionTreeRegressor(max_depth=self.tree_max_depth, min_samples_split=5, min_samples_leaf=5)
        else:
            base = model
        base.fit(x, -self.compute_gradient(x, y, var))
        self.model.append((self.shrinkage, base))
        return self

    def drop_model(self):
        self.model = self.model[:-1]

    def predict(self, x):
        if len(self.model) == 0:
            return np.zeros(x.shape[0])
        else:
            predict = [i*j.predict(x) for (i, j) in self.model]
            return np.sum(np.asarray(predict), axis=0)


    def compute_gradient(self, x, y, var):
        #var is the value of O(x) given x
        return -np.exp(-var)*(y - self.predict(x))

class Var():
    #A class for the variance estimation for GBM
    def __init__(self, x, tree_max_depth):
        #x is a numpy array, the input of u
        self.x = x
        self.gradient = np.zeros(x.shape[0])
        self.output = np.zeros(x.shape[0])
        self.tree_max_depth = tree_max_depth
        self.model = []
        self.shrinkage = 0.1

    def fit(self, x, y, mean, model = None):
        #based on h(x), find one base estimator to -gm
        #now use a base tree learner

        #Create a new tree
        if model == None:
            base = DecisionTreeRegressor(max_depth=self.tree_max_depth, min_samples_split=2, min_samples_leaf=5)
        else:
            base = model
        base.fit(x, -self.compute_gradient(x, y, mean))
        self.model.append((self.shrinkage, base))
        return self

    def drop_model(self):
        self.model = self.model[:-1]

    def predict(self, x):
        if len(self.model) == 0:
            return np.zeros(x.shape[0])
        else:
            predict = [i*j.predict(x) for (i, j) in self.model]
            return np.sum(np.asarray(predict), axis=0)


    def compute_gradient(self, x, y, mean):
        #var is the value of O(x) given x
        return 0.5-((y - mean)**2)*np.exp(-self.predict(x))/2.0


#TRY: Design a new base learner to fit the variance part
#Since the variacnace part should be more stable/smoothy. Try to reduce the functional complexity of variance.
#Insted of using trees as base learner, try a liner model.
#PROVED NOT WORKING
class expand_linear_base():
    def __init__(self, d_start, d_end):
        self.d_start = d_start
        self.d_end = d_end
        self.model = LinearRegression()

    def fit(self, x, y):
        self.model.fit(expand(np.squeeze(x), self.d_start, self.d_end), y)

    def predict(self, x):
        return self.model.predict(expand(np.squeeze(x), self.d_start, self.d_end))




def nll_loss(x, y, mean_func, var_func):
    return np.sum(var_func(x) + (y - mean_func(x))**2/np.exp(var_func(x)))/2


def viz(x_train, x_test, y_train, y_test, mean_func, var_func):
    """
    visualize the data, with classifier embedded
    """
    x = np.squeeze(np.concatenate((x_train, x_test)))
    y = np.concatenate((y_train, y_test))
    color = ['r']*len(x_train)+['b']*len(y_train)
    plt.scatter(x, y, c=color, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data fitted with GBM')

    #visualize the model in plot
    #plot mean line
    x_c = np.array(range(500))/500.
    mean_c = mean_func(np.expand_dims(x_c, axis=1))
    variance_c = np.exp(var_func(np.expand_dims(x_c, axis=1)))

    #plot variance line
    plt.plot(x_c, mean_c, 'b')
    plt.plot(x_c, mean_c+np.sqrt(variance_c),'r')
    plt.plot(x_c, mean_c-np.sqrt(variance_c), 'r')
    plt.show()

"""
    #print the nll value
    mean_test = mean_func(x_test)
    var_test = var_func(x_test)
    #     mean_test, var_test = y_pred[0], y_pred[1]
    nll = np.sum(compute_neglog(y_test[i], mean_test[i], var_test[i]) for i in range(len(y_test)))
    print "NNL on test data: %r" %(nll)
"""
def gbm_train(x_train, x_test, y_train, y_test):

    np.random.seed(seed=10)
    u_estimate = U(x_train, tree_max_depth=3)
    var_estimate = Var(x_train, tree_max_depth=3)

    #use another variance base estimator
    var_model = expand_linear_base(d_start=0, d_end=3)
    train_loss, var_loss = 100000., 100000.

    for m in range(67):
        x_train_1, x_train_2, y_train_1, y_train_2 = train_test_split(x_train, y_train, test_size=0.5, random_state=m)

        #u_estimate.fit(x_train, y_train, var_estimate.predict(x_train), model=var_model)
        #var_estimate.fit(x_train, y_train, u_estimate.predict(x_train))

        u_estimate.fit(x_train_1, y_train_1, var_estimate.predict(x_train_1), model=None)
        while nll_loss(x_train, y_train, u_estimate.predict, var_estimate.predict) > train_loss:
            u_estimate.drop_model()
            print "Using line search, re-train"
            u_estimate.shrinkage = u_estimate.shrinkage/2.
            u_estimate.fit(x_train_1, y_train_1, var_estimate.predict(x_train_1), model=None)


        var_estimate.fit(x_train_2, y_train_2, u_estimate.predict(x_train_2))
        while nll_loss(x_train, y_train, u_estimate.predict, var_estimate.predict) > train_loss:
            var_estimate.drop_model()
            print "Using line search, re-train"
            var_estimate.shrinkage = var_estimate.shrinkage/2.
            u_estimate.fit(x_train_2, y_train_2, var_estimate.predict(x_train_2), model=None)



        train_loss = nll_loss(x_train, y_train, u_estimate.predict, var_estimate.predict)
        var_loss = nll_loss(x_test, y_test, u_estimate.predict, var_estimate.predict)
        print "Iter %r, Train Loss: %.3f, Validation Loss: %.3f" %(m, train_loss, var_loss)

        #visualize every 50 iter

        #if m%25 == 0:
            #viz(x_train, x_test, y_train, y_test, u_estimate.predict, var_estimate.predict)
            #print np.exp(var_estimate.predict(x_train))
    viz(x_train, x_test, y_train, y_test, u_estimate.predict, var_estimate.predict)
    return u_estimate, var_estimate


def viz_par(mean_func, var_func, U_est, V_est):
    #visualize the parameters
    #visualize par
    x_c = np.array(range(500))/500.
    mean_x = mean_func(x_c)
    var_x = var_func(x_c)

    mean_test = [U_est.predict(i) for i in x_c]
    var_test = np.exp(np.asarray([V_est.predict(i) for i in x_c]))

    plt.plot(x_c, mean_x, 'b')
    plt.plot(x_c, mean_test, 'b--')
    plt.plot(x_c, var_x, 'r')
    plt.plot(x_c, var_test, 'r--')
    plt.xlabel('x')
    plt.ylabel('parameters')
    plt.legend(['mean(x)', 'Prediction of mean', 'std_err(x)', 'Prediction of std_err'], loc=3)
    plt.show()


def main():
    x_train, x_test, y_train, y_test = read_data('func_change_cor_3.csv')
    x_train, x_test = np.expand_dims(x_train, axis=1), np.expand_dims(x_test, axis=1)

    u_estimate, var_estimate = gbm_train(x_train, x_test, y_train, y_test)
    #viz_par(sin_scale, log_scale, u_estimate, var_estimate)
    #viz_par(lambda x: np.sin(2.5*x*3.14)*np.sin(1.5*x*3.14), lambda x: 0.01+0.25*(1-np.sin(2.5*x*3.14))**2, u_estimate, var_estimate)

    l1, l2 = generate_random()
    viz_par(l1, l2, u_estimate, var_estimate)


main()