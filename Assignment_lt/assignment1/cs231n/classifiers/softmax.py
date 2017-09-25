import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  - loss as single float(数)
  - gradient with respect to weights W; an array of same shape as W（二维数组3073*10）
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
  #loss   and  gradient
  num_train=X.shape[0]
  num_classes=W.shape[1]
  for i in range(num_train):
      scores=X[i].dot(W)  #1 by C
      shift_scores=scores-max(scores)  # # 找到最大值然后减去，这样是为了防止后面的操作会出现数值上的一些偏差
      loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
      loss+=loss_i
      for j in range(num_classes):
          softmax_output= np.exp(shift_scores[j])/(sum(np.exp(shift_scores)))
          if j==y[i]:
              dW[:,j]+=(-1 + softmax_output) *X[i] 
          else: 
              dW[:,j] += softmax_output *X[i] 
  loss /=num_train
  loss += 0.5*reg*np.sum(W*W)
  dW =dW/num_train + reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train=X.shape[0]
  scores=X.dot(W)   # N by C
  shift_scores=scores-np.max(scores,axis=1).reshape(-1,1)  #为了避免数值计算出现不稳定
  softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)
  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))  #所有样本损失的求和
  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  
  dS = softmax_output.copy()   #N by C
  dS[range(num_train), list(y)] += -1
  dW = (X.T).dot(dS)   #N by D   N by C
  dW = dW/num_train + reg* W 
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

