'''
Created on Jul 4, 2016
ADAptive LInear NEuron (Adaline).
In ADALINE the weights are updated based on a linear activation function
(the identity function) rather than a unit step function like in the perceptron.

@author: jeromeboyer
'''
from numpy.random import seed

import numpy as np


class AdalineGradiantDescent(object):
    '''
    Adaptive Linear Neuron Classifier
    '''

    def __init__(self, eta=0.01,nbOfIteration=10):
        '''
        Constructor: eta is the learning rate between 0 and 1. The # of iteration
        to train by going n times over the training set
        '''
        self.eta=eta
        self.nbOfIteration=nbOfIteration

    def fit(self,X,y):
        ''' Fit the training data given within the matrix X of the samples/features and y the matching classes
        Recompute the weight if there is a delta between the predicted value and the expected value
        X : {array-like}, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] of class labels
        '''
        # the number of weights = nb of features + 1.
        self.weights=np.zeros(1+X.shape[1])
        self.costs=[]
        for _ in range(self.nbOfIteration):
            output = self.netInput(X)
            errors = (y - output)
            # calculate the gradient based on the whole training dataset. Use the matrix vector *
            # between the feature matrix and the error vector
            self.weights[1:] += self.eta * X.T.dot( errors)
            self.weights[0]  += self.eta * errors.sum()
            cost = (errors** 2).sum() / 2.0
            # keep cost computed at each iteration to assess convergence.
            self.costs.append(cost)
        return self

    def netInput(self,X):
        # compute z = sum(x(i) * w(i)) for i from 1 to n, add the threshold
        return np.dot(X,self.weights[1:]) + self.weights[0]

    def activation( self, X):
        """ Compute linear activation"""
        return self.netInput( X)

    def predict(self,X):
        # unit step function. Compute the net input and compares it a threshold
        return np.where(self.activation(X)>=0.0,1,-1)


class AdalineStockaticGradiantDescent(object):
    def __init__(self, eta=0.01,nbOfIteration=10,shuffle=True,randomState=None):
        '''
        Constructor
        '''
        self.eta=eta
        self.nbOfIteration=nbOfIteration
        self.shuffle=shuffle
        self.weightInitialized=False
        if randomState:
            seed(randomState)

    def fit(self,X,y):
        self.initWeights(X.shape[1])
        self.costs=[]
        for _ in range(self.nbOfIteration):
            if self.shuffle:
                X, y = self.shuffleData( X, y)
            cost = []
            for xi, target in zip( X, y):
                cost.append( self.updateWeights( xi, target))
            averageCost = sum(cost)/ len( y)
            self.costs.append( averageCost)
        return self

    def initWeights(self,m):
        # init weigths with zero
        self.weights=np.zeros(1+m)
        self.weightInitialized=True

    def updateWeights(self,xi,target):
        """ update the weights incrementally for each training sample.
        xi a sample of th feature: a row of the matrix
        """
        output = self.netInput( xi)
        error = (target - output)
        self.weights[1:] += self.eta * xi.dot( error)
        self.weights[0] += self.eta * error
        cost = (error** 2)/ 2.0
        return cost

    def shuffleData(self,X,y):
        """ Shuffle training data"""
        r = np.random.permutation( len( y))
        return X[ r], y[ r]

    def netInput(self,X):
        # compute z = sum(x(i) * w(i)) for i from 1 to n, add the threshold
        return np.dot(X,self.weights[1:]) + self.weights[0]

    def activation( self, X):
        """ Compute linear activation"""
        return self.netInput( X)

    def predict(self,X):
        # unit step function. Compute the net input and compares it a threshold
        return np.where(self.activation(X)>=0.0,1,-1)

    def partialFit(self,X,y):
        # Fit without init the weights
        if not self.weightInitialized:
            self.initWeights(X.shape[1])
        if y.ravel.shape[0] > 1:
            for xi, target in zip( X, y):
                self.updateWeights( xi, target)
        else:
            self.updateWeights(X,y)
        return self
if __name__ == "__main__":
    pass
