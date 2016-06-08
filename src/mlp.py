#Implement MLP for density estimation

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from data_generate import sin_scale, log_scale, generate_random
from utils import read_data, expand, compute_neglog

class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.
        '''
        # Retrieve the input and output dimensionality based on W's initialization
        n_output, n_input = W_init.shape
        # Make sure b is n_output in size
        assert b_init.shape == (n_output,)

        self.W = theano.shared(value=W_init.astype(theano.config.floatX),

                               name='W',
                               borrow=True)
        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               broadcastable=(False, True))
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]

    def output(self, x):
        lin_output = T.dot(self.W, x) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation is None else self.activation(lin_output))



class MLP(object):
    def __init__(self, W_init, b_init, activations):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)

        # Initialize lists of layers
        self.layers = []
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        '''
        Compute the MLP's output given an input

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def nll_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y
        '''
        mean, var = self.output(x)[0], self.output(x)[1]
        error = var/2 + (mean - y)**2*T.exp(-var)/2
        return T.sum(error)


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates




def mlp_train(x_train, x_test, y_train, y_test):
    print x_train.shape, x_test.shape
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'

    layer_sizes = [x_train.shape[0], 10, 2]
    # Set initial parameter values
    W_init = []
    b_init = []
    activations = []
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):

        W_init.append(np.random.randn(n_output, n_input)+5)
        b_init.append(np.zeros(n_output))
        activations.append(T.nnet.sigmoid)
    #The last layer should not be sigmoid
    activations[-1] = None
    mlp = MLP(W_init, b_init, activations)

    # Create Theano variables for the MLP input
    mlp_input = T.matrix('mlp_input')
    # ... and the desired output
    mlp_target = T.vector('mlp_target')
    n_iter = T.scalar('n_iter')

    learning_rate = 0.1
    momentum = 0.0
    train_cost=1000000

    # Create a function for computing the cost of the network given an input
    cost = mlp.nll_error(mlp_input, mlp_target)
    train = theano.function([mlp_input, mlp_target, n_iter], cost,
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate/n_iter, momentum))
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

    iteration = 0
    div = 1
    max_iteration = 30000


    while iteration < max_iteration:
        #Learning rate updated every 10 epoches
        """
        if iteration < 1000:
            div = iteration/75+1
        else:
            div = 10
        """

        #line search
        init_value = []
        for param in mlp.params:
            init_value.append(param.get_value())

        train(x_train, y_train, div)
        while mlp.nll_error(x_train, y_train).eval() >= train_cost:
            print "line search, change learning rate"
            div = div+1
            i=0
            for param in mlp.params:
                param.set_value(init_value[i])
                i+=1
            train(x_train, y_train, div)

        train_cost = mlp.nll_error(x_train, y_train).eval()
        val_cost = mlp.nll_error(x_test, y_test).eval()

        # Get the current network output for all points in the training set
        current_output = mlp_output(x_train)
        print "Epoch: %r Train Cost: %.3f, Val cost: %.3f" %(iteration, train_cost, val_cost)
        iteration += 1

    return mlp

##implement line search for step size


def viz_clf(x_train, x_test, y_train, y_test, mean_func, var_func, number, mlp):
    """
    visualize the data, with classifier embedded
    """
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    color = ['r']*len(x_train)+['b']*len(y_train)
    plt.scatter(x, y, c=color, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data fitted with MLP parametric variance')

    #visualize the model in plot
    #plot mean line
    x_c = np.array(range(500))/500.
    y_c = mlp.output(expand(x_c,1,number).T).eval()
    #print y_c
    mean_c = y_c[0]
    variance_c = np.exp(y_c[1])

    #plot variance line
    plt.plot(x_c, mean_c, 'b')
    plt.plot(x_c, mean_c+np.sqrt(variance_c),'r')
    plt.plot(x_c, mean_c-np.sqrt(variance_c), 'r')
    plt.show()


    #visualize par
    mean_x = mean_func(x_c)
    var_x = var_func(x_c)
    y_pred = mlp.output(expand(x_c,1,number).T).eval()
    mean_test, var_test = y_pred[0], np.exp(y_pred[1])

    plt.plot(x_c, mean_x, 'b')
    plt.plot(x_c, mean_test, 'b--')
    plt.plot(x_c, var_x, 'r')
    plt.plot(x_c, var_test, 'r--')
    plt.xlabel('x')
    plt.ylabel('parameters')
    plt.legend(['mean(x)', 'Prediction of mean', 'std_err(x)', 'Prediction of std_err'], loc=3)
    plt.show()

    #nll = np.sum(compute_neglog(y_test[i], mean_test[i], var_test[i]) for i in range(len(y_test)))
    #print "NNL on test data: %r" %(nll)


def main():
    x_train, x_test, y_train, y_test = read_data('func_change_cor_3.csv')
    np.random.seed(seed=10)
    s_end = 3
    targetFn, varFn = generate_random()
    mlp = mlp_train(expand(x_train,1,s_end).T, expand(x_test,1,s_end).T, y_train, y_test)
    #viz_clf(x_train, x_test, y_train, y_test, lambda x: np.sin(2.5*x*3.14)*np.sin(1.5*x*3.14), lambda x: 0.01+0.25*(1-np.sin(2.5*x*3.14))**2,s_end, mlp)
    viz_clf(x_train, x_test, y_train, y_test, targetFn, varFn, s_end, mlp)


main()


class BankAccount:
    """ Class definition modeling the behavior of a simple bank account """

    def __init__(self, initial_balance):
        """Creates an account with the given balance."""
        self.balance = initial_balance
        self.fees = 0
    def deposit(self, amount):
        """Deposits the amount into the account."""
        self.balance += amount
    def withdraw(self, amount):
        """
        Withdraws the amount from the account.  Each withdrawal resulting in a
        negative balance also deducts a penalty fee of 5 dollars from the balance.
        """
        self.balance-=amount
        if self.balance<0:
            self.balance-=5
            self.fees+=5
    def get_balance(self):
        """Returns the current balance in the account."""
        return self.balance
    def get_fees(self):
        """Returns the total fees ever deducted from the account."""
        return self.fees