from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
from math import log, exp

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Multiply the weight with the input then add the bias.
        # The weight matrix has H neurons, each with D separate weights,
        # where each weight corresponds to a dimension in the input.
        # The resulting matrix is N x H (one set of weights per neuron for each input)
        input_to_relu = np.matmul(X, W1) + b1

        # Define the ReLU neuron activation function,
        # i.e., negative values are zeroed, positive values are retained as is
        relu = lambda val: max(0, val)

        # Apply the ReLU activation function to each value
        # on a row by row basis,
        # i.e., generate a new array for each row in the original scores matrix
        # where the function is applied over each value
        output_of_relu = np.array([[relu(val) for val in row] for row in input_to_relu])

        # Multiply the resulting score matrix with the second layer's weights, then add the biases.
        # The weight matrix has H neurons but now each group of 10 are linked to a distinct output class.
        # The output matrix here is N x C (one set of class predictions per input)
        scores = np.matmul(output_of_relu, W2) + b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Note: the output logits are not normalized using a softmax function
        # so it will be applied now
        def softmax(row):
            # Convert the row's scores into e^(score)
            row = [exp(score) for score in row]

            # Find the sum of the row to normalize all the values
            sum_of_row = sum(row)

            # Return the normalized row
            return [score / sum_of_row for score in row]
        probabilities = [softmax(row) for row in scores]

        # Set a loss variable for sum of loss from data only
        data_loss = 0

        # Identify each input data point by its index to go through each
        for datapoint_index in range(0, N):
            correct_class = y[datapoint_index]

            # The softmax loss for each input data point is -log(P(y))
            # where P(y) is the probability we generate for the correct class.
            # The total loss requires the sum of all loss for each data point.
            data_loss += -log(probabilities[datapoint_index][correct_class])

        # The total data loss is averaged over the number of data points
        data_loss /= N

        # The regularization method that we are using is L2 regularization (i.e., ridge regularization).
        # The loss for the whole model is the sum of the squares of the weights, both W1 and W2.
        # Note: here we do not add the 0.5 regularization multiplier.
        reg_loss_1 = reg * np.sum(W1 * W1)
        reg_loss_2 = reg * np.sum(W2 * W2)

        # The total loss is the sum of data loss and all the regularization losses
        loss = data_loss + reg_loss_1 + reg_loss_2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # -------------- W2 --------------

        # The derivative of the sum of losses with respect to the scores (logits) before softmax
        # is ((the probabilities generated by the softmax function) - 1) / (number of inputs).
        # The -1 only applies to the correct class's softmax probability.
        # This is because the full crose entropy error function also depends on 
        # the other logits (despite they are zeroed out), which in the full derivation will cause this
        # after multiplying with the softmax derivative.
        dLoss_dScore = []
        for datapoint_index in range(N):
            row = []
            for class_index in range(len(probabilities[datapoint_index])):
                probability = probabilities[datapoint_index][class_index]
                row += [(probability - 1) / N if y[datapoint_index] == class_index else probability / N,]
            dLoss_dScore += [row,]

        # The derivative between the scores (logits) and the second layer weights W2
        # is the inputs themselves (i.e., the outputs from ReLU neurons) because the second layer is a linear combination.
        # From the size of the matrices, we see outputs from the ReLU to be 5 x 10
        # and dLoss_dScore to be 5 x 3, and we want the resulting matrix to be 10 x 3.
        # Hence, we take the transpose of the outputs from the ReLU to match the dimensionality.
        # An interpretation is that we are multiplying the neuron value for input i with the dLoss_dScore for input i's class j,
        # then summing it across all inputs, to generate the differential for the weight given to the first neuron
        # for the class j. 
        dL_dW2 = np.dot(output_of_relu.T, dLoss_dScore)

        # The differential of regularization loss with respect to each weight is just 2 * weight * regularization constant.
        # Apply it to dL_dW2 here
        dL_dW2 += 2 * reg * W2
        grads['W2'] = dL_dW2

        # -------------- b2 --------------

        # Differential between the scores (logits) and the bias b2 is just 1 (because it's just addition),
        # hence, it essentially copies the differential of the loss with respect to the score (logit) values.
        # However, since we have N different data points, we need to sum their individual differentials for each class.
        grads['b2'] = []
        for class_index in range(len(dLoss_dScore[0])):
            differential_sum = 0
            for row in dLoss_dScore:
                differential_sum += row[class_index]
            grads['b2'] += [differential_sum,]

        # -------------- W1 --------------

        # Differential of the output of the ReLU with respect to the logits is just the W2 weights.
        # So the dLoss_dReluOut is just (W2 weights) * (dLoss_dScore).
        # The interpretation of the transpose is that we multiply the loss differential per class (per data point)
        # with the weight for each class per neuron, which generates
        # how much loss changes given shifts in each neuron's value for each data point.
        dLoss_dReluOut = np.dot(dLoss_dScore, W2.T)

        # Differential before the ReLU is just dLoss_dReluOut but zeroed for the ones whose input to the ReLU is <= 0
        dLoss_dReluIn = dLoss_dReluOut.copy()
        for row in range(N):
            for column in range(len(dLoss_dReluIn[0])):
                if input_to_relu[row][column] <= 0:
                    dLoss_dReluIn[row][column] = 0
        
        # Similar to W2, for W1, the differential between inputs to the ReLU and the weights are the initial inputs
        dL_dW1 = np.dot(X.T, dLoss_dReluIn)

        # Again, similar to W2, we also need to add the regularization loss' differential.
        # They have the same formula, but instead of W2, it is W1.
        dL_dW1 += 2 * reg * W1
        grads['W1'] = dL_dW1

        # -------------- b1 --------------

        # The procedure here is basically the same as that of b2's,
        # the only difference being, it now copies the differential of the loss with respect to the ReLU inputs.
        grads['b1'] = []
        for neuron_index in range(len(dLoss_dReluIn[0])):
            differential_sum = 0
            for row in dLoss_dReluIn:
                differential_sum += row[neuron_index]
            grads['b1'] += [differential_sum,]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
