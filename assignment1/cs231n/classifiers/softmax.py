import numpy as np
from random import shuffle

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

  data_loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)

    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores)

    probs = exp_scores / sum_exp_scores

    correct_logprobs = -np.log(probs[y[i]])

    data_loss += correct_logprobs

    probs[y[i]] -= 1
    for k in range(num_classes):
      dW[:, k] += probs[k] * X[i]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  data_loss /= num_train
  reg_loss = 0.5 * reg * np.sum(W * W)

  loss = data_loss + reg_loss


  dW /= num_train
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)

  scores -= np.max(scores, axis=1, keepdims=True)

  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores)

  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  correct_logprobs = -np.log(probs[range(num_train),y])

  # compute the loss: average cross-entropy loss and regularization
  data_loss = np.sum(correct_logprobs)/num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss

  dscores = probs
  dscores[range(num_train),y] -= 1
  dscores /= num_train

  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  dW += reg*W # don't forget the regularization gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

