#Implement MLP for density estimation

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
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
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
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

    learning_rate = 0.005
    momentum = 0.5

    # Create a function for computing the cost of the network given an input
    cost = mlp.nll_error(mlp_input, mlp_target)
    train = theano.function([mlp_input, mlp_target, n_iter], cost,
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate/n_iter, momentum))
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

    iteration = 0
    div = 1
    max_iteration = 10000

    while iteration < max_iteration:
        #Learning rate updated every 10 epoches
        if iteration < 1000:
            div = iteration/75+1
        else:
            div = 10
        train_cost = train(x_train, y_train, div)
        val_cost = mlp.nll_error(x_test, y_test).eval()

        # Get the current network output for all points in the training set
        current_output = mlp_output(x_train)
        print "Epoch: %r Train Cost: %.3f, Val cost: %.3f" %(iteration, train_cost, val_cost)
        iteration += 1

    return mlp

#

def viz_clf(x_train, x_test, y_train, y_test, mlp):
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
    y_c = mlp.output(expand(x_c,1,1).T).eval()
    #print y_c
    mean_c = y_c[0]
    variance_c = np.exp(y_c[1])

    #plot variance line
    plt.plot(x_c, mean_c, 'b')
    plt.plot(x_c, mean_c+np.sqrt(variance_c),'r')
    plt.plot(x_c, mean_c-np.sqrt(variance_c), 'r')


    #print the nll value
    y_pred = mlp.output(expand(x_test,1,1).T).eval()
    mean_test, var_test = y_pred[0], y_pred[1]
    nll = np.sum(compute_neglog(y_test[i], mean_test[i], var_test[i]) for i in range(len(y_test)))
    print "NNL on test data: %r" %(nll)

    plt.show()

def main():
    x_train, x_test, y_train, y_test = read_data('func_change_cor.csv')
    mlp = mlp_train(expand(x_train,1,1).T, expand(x_test,1,1).T, y_train, y_test)

    viz_clf(x_train, x_test, y_train, y_test, mlp)


main()