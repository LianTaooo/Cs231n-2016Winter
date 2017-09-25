import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]  #样本的类别数
  num_train = X.shape[0]    #训练样本数
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]  #正确类对应的得分
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin   #多个样本的和
        dW[:,j] += X[i].T   #多个样本的和 （由margin公式可求得）
        dW[:,y[i]] += -X[i].T 
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train
  # Add regularization to the loss.
  loss += 0.5* reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #loss
  scores = X.dot(W) #N by C
  num_train=X.shape[0]
  num_class=W.shape[1]
  scores_correct=scores[np.arange(num_train),y]  #在scores矩阵中取出正确的值 1 by N
  scores_correct=np.reshape(scores_correct,(num_train,1))    # N by 1
  margins = scores-scores_correct +1.0 #N by C
  margins[np.arange(num_train),y]=0.0  #分类正确时损失为0
  margins[margins<0]=0.0
  loss+=np.sum(margins)/num_train  #所有数加起来取均值
  #加上正则化
  loss+=0.5*reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #gradient
  margins[margins>0]=1.0    #  1.为了计算有代价的类数    2.有代价类置为1
  row_sum=np.sum(margins,axis=1)  #1 by N   #每一个样本有代价值的类数
  margins[np.arange(num_train),y]=-row_sum   #该类数放到正确分类位置
  dW+=np.dot(X.T,margins)/num_train + reg*W   #D by C    np.dot(X.T,margins)：每一项都为 x每一项 的倍数
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
