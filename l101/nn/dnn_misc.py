import numpy as np

class linear_layer:

    """
        The linear (affine/fully-connected) module.

        It is built up with two arguments:
        - input_D: the dimensionality of the input example/instance of the forward pass
        - output_D: the dimensionality of the output example/instance of the forward pass

        It has two learnable parameters:
        - self.params['W']: the W matrix (numpy array) of shape input_D-by-output_D
        - self.params['b']: the b vector (numpy array) of shape 1-by-output_D

        It will record the partial derivatives of loss w.r.t. self.params['W'] and self.params['b'] in:
        - self.gradient['W']: input_D-by-output_D numpy array
        - self.gradient['b']: 1-by-output_D numpy array
    """

    def __init__(self, input_D, output_D):
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))

        self.gradient = dict()
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):

        """
            The forward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, where each 'row' is an input example/instance (i.e., X[i], where i = 1,...,N).
                The mini-batch size is N.

            Return:
            - forward_output: A N-by-output_D numpy array, where each 'row' is an output example/instance.
        """
        
        return np.dot(X, self.params['W']) + self.params['b']

    def backward(self, X, grad):

        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, the input to the forward pass.
            - grad: A N-by-output_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. forward_output[i].

            Return:
            - backward_output: A N-by-input_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. X[i].
        """
        
        self.gradient['W'] = np.dot(np.transpose(X), grad)
        self.gradient['b'] = np.dot(np.transpose(grad), np.ones(X.shape[0]))
        backward_output = np.dot(grad, np.transpose(self.params['W']))
        
        return backward_output


class relu:

    """
        The relu (rectified linear unit) module.
    """

    def __init__(self):
        donothing = 1
    def forward(self, X):

        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """

        return np.maximum(0, X)

    def backward(self, X, grad):

        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in  X.
        """

        backward_output = np.copy(grad)
        backward_output[X < 0] = 0
        return backward_output

class softmax_cross_entropy:

    """
        Module that computes softmax cross entropy loss.
        self.prob is an N x K matrix that can be used to store the probabilites of N samples for each of the K classes(calculated in the forward pass).
        self.expand_Y is an N x K matrix that can be used to store the one-hot encoding of the true label Y for the N training samples.
    """

    def __init__(self):
        self.expand_Y = None
        self.prob = None

    def forward(self, X, Y):

        """
            The forward pass of the softmax_cross_entropy module.

            Input:
            - X: A numpy array of of shape N_by_K where N is the mini-batch size and K is the number of classes.
            - Y: A numpy array of shape N_by_1 which has true labels for the N training samples in the minibatch.

            Return:
            - forward_output: A single number that indicates the cross entropy loss between softmax(X) and Y
        """

        # compute softmax        
        softmax = []
        for x in X:
            expa = np.exp(x - np.max(x))
            softmax.append(expa / np.sum(expa))
        softmax = np.array(softmax)
        self.prob = softmax
        
        # one-hot encoding
        n = softmax.shape[1]
        newY = []
        for y_list in Y:
            new_y_list = []
            old_val = y_list[0]
            for i in range(int(old_val)):
                new_y_list.append(0)
            new_y_list.append(1)
            for i in range(int(n - old_val - 1)):
                new_y_list.append(0)
            newY.append(new_y_list)
            
        self.expand_Y = np.array(newY)
        
        # computing cross entropy loss
        n = X.shape[0]
        loss = - 1 / n * np.sum(np.diagonal(np.dot(np.transpose(self.expand_Y), np.log(self.prob))))
        
        return loss

    def backward(self, X, Y):
        """
            The backward pass of the softmax_cross_entropy module.

            Input:
            - X: A numpy array of of shape N_by_K where N is the mini-batch size and K is the number of classes.
            - Y: A numpy array of shape N_by_1 which has true labels for the N training samples in the minibatch.

            Return:
            - backward_output: A matrix of shape N_by_K where N is the size of the minibatch and K is the number of classes.
        """
        
        n = X.shape[0]
        
        return (self.prob - self.expand_Y) / n


class flatten_layer:

    def __init__(self):
        self.size = None

    def forward(self, X):
        self.size = X.shape
        out_forward = X.reshape(X.shape[0], -1)

        return out_forward

    def backward(self, X, grad):
        out_backward = grad.reshape(self.size)

        return out_backward

### Momentum ###
def add_momentum(model):
    momentum = dict()
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                momentum[module_name + '_' + key] = np.zeros(module.gradient[key].shape)
    return momentum