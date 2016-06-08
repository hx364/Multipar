__author__ = 'how'
#Implement MLP for density estimation
#Use conjugate gradient descent
class Layer(object):
    def __init__(self, W_init, b_init, activation):
        
        n_output, n_input = W_init.shape
        assert b_init.shape == (n_output,)
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               broadcastable=(False, True))
        self.activation = activation
        self.params = [self.W, self.b]

    def output(self, x):
        lin_output = T.dot(self.W, x) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))

class cg_MLP(object):
    def __init__(self, input, n_input, n_hidden, n_output):
        
        #initialize the params
        W_init_1 = np.random.randn(n_hidden, n_input)



class MLP(object):
    def __init__(self, W_init, b_init, activations):
        assert len(W_init) == len(b_init) == len(activations)
        self.layers = []
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def nll_error(self, x, y):
        #Compute modified nll error
        mean, var = self.output(x)[0], self.output(x)[1]
        error = var/2 + (mean - y)**2*T.exp(-var)/2
        return T.sum(error)

def mlp_cg_optimization_train(x_train, x_test, y_train, y_test):
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

    # Create a function for computing the cost of the network given an input
    cost = mlp.nll_error(mlp_input, mlp_target)
    train = theano.function([mlp_input, mlp_target, n_iter], cost,
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate/n_iter, momentum))
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

    
    
    #create a function that computes the cost on traing set
    
    

