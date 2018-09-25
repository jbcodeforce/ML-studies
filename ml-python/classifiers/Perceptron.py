'''
Created on Jun 26, 2016
Based on Python Machine Learning book from
Page 25
@author: jeromeboyer
'''

import numpy as np


class Perceptron(object):
    '''
    Perceptron classifier is based on Rosenblatt's algorithm unit step function.
    Based on the human neuron model, Frank Rosenblatt proposed an algorithm that would automatically learn the optimal weight coefficients that are then multiplied with the input features in order to make the decision of whether a neuron fires or not. In the context of supervised learning and classification, such an algorithm could then be used to predict if a sample belonged to one class or the other.
    '''
    def __init__(self, eta=0.01,nbOfIteration=10):
        '''
        Constructor: eta is the learning rate between 0 and 1. The # of iteration
        to compute the weights by going n loop over the training set
        '''
        self.eta=eta
        self.nbOfIteration=nbOfIteration

    def fit(self,X,y):
        ''' Fit the training data given the matrix X of the samples/features and y the matching classes
        Recompute the weight if there is a delta between the predicted value and the expected value
        X : {array-like}, shape = [n_samples, p_features]
        y : array-like, shape = [n_samples] of class labels
        '''
        self.weights=np.zeros(1+X.shape[1])
        # list of misclassifications during the epochs
        self.errors=[]

        for _ in range(self.nbOfIteration):
            error=0
            # iterate over the 'sample' row of X and y, stop on shortest list between X & y(the use of zip)
            for xi,target in zip(X,y):
                # update the weights according to the perceptron learning rule
                update=self.eta*(target - self.predict(xi))
                self.weights[1:]+=update * xi
                self.weights[0]+=update
                error+=int(update != 0.0)
            self.errors.append(error)
        return self

    def netInput(self,X):
        # calculate the vector dot product wT * x, or equivalent to
        # compute z = sum(x(i) * w(i)) for i from 1 to n, add the threshold
        return np.dot(X,self.weights[1:]) + self.weights[0]

    def predict(self,X):
        # predict the class label
        # unit step function: return -1 or 1 depending of the
        # net input compared to a threshold
        return np.where(self.netInput(X)>=0.0,1,-1)



if __name__ == "__main__":
    '''
        The way to use the perceptron: Create an instance by specifying the eta coefficient
        and the number of epochs (passes over the training set) to perform
    '''
    p=Perceptron(0.01,10)

    #p.fit(X,y)
    pass
