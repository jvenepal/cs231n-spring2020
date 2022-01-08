from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, num_dim = X.shape
    for i in range(num_train):
        scores = X[i].dot(W) # (C,) np.exp(scores) can be arbitarly large, causing numeric
        scores -= scores.max() # instability. Now, the max value is 0. Both give same result.
        probs = np.exp(scores) / np.sum(np.exp(scores)) # (C,)
        correct_class_prob = probs[y[i]] # (1,)
        loss += -np.log(correct_class_prob)
        
        # jash: gradient calculation is based on below links:
        # 1. https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # 2. https://deepnotes.io/softmax-crossentropy
        # data flow is: scores->(softmax)->probs->(cross-entory)->loss
        # derivative of the cross-entropy loss wrt. scores is (probs - y), where
        # y is a one-hot verctor of labels
        dscores = probs.reshape(1, -1) # (1, C)
        ys = np.zeros_like(dscores) # (1, C)
        ys[:, y[i]] = 1
        dscores -= ys
        # dloss/dw = dloss/dscores * dscores/dw
        # dscores/dw is X[i] itself of shape (W,)
        # since dw is of shape (W, C), X[i].T.dot(dscores) gives dw
        dW += (X[i][np.newaxis, :].T).dot(dscores)
        
    loss /= num_train # since we sum the loss over all training samples, take mean
    dW /= num_train
    loss += reg * np.sum(W * W) # reg. loss
    dW += 2 * reg * W # derivative of reg. loss
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # loss
    num_train, dim = X.shape
    scores = X.dot(W) # (N, C)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, np.newaxis] # (N, C)
#     assert not np.all(probs.sum(axis=1)), "probs should sum to 1 along axis=1"
    correct_class_probs = probs[range(num_train), y] # (N,)
    loss = np.mean(-np.log(correct_class_probs))
    loss += reg * np.sum(W * W) # add reg. loss
    
    # gradients, see softmax_loss_naive for details
    ys = np.zeros_like(probs) # (N, C)
    ys[range(num_train), y] = 1 # one-hot vector of true labels (N, C)
    dscores = probs - ys # (N, C)
    dW = X.T.dot(dscores) # (D, N) * (N, C) = (D, C)
    dW /= num_train # mean
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
