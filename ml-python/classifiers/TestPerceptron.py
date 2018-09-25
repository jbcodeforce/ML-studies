'''
Created on Jun 24, 2016

@author: jeromeboyer
'''
import unittest

from matplotlib.colors import ListedColormap

from Perceptron import Perceptron
import Tool as tool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TestPerceptron(unittest.TestCase):

    def loadIrisData(self):
        # load the two flower classes Setosa and Versicolor from the Iris dataset.
        df=pd.read_csv('../../data/iris_data.txt',header=None)
        return df

    def testLoadIrisData(self):
        print("\n\n>> Test load iris data - the last 5 rows of the iris data are:\n")
        df=self.loadIrisData()
        print(df.tail())
        pass

    def testPlotIrisData(self):
        print("\n\n>> Plot the iris data for 100 records, x axe is for sepal, y for petal")
        print(" ################################################### ")
        df=self.loadIrisData()
        # build the label list for the first 100 records, label is in 5th column (idx=4)
        y=df.iloc[0:100,4].values
        # for neural network dual class label should be -1 or 1, so modify y
        y = np.where( y == 'Iris-setosa', -1, 1)
        # take the columns 0,1: sepal and petal length
        X=df.iloc[0:100,[0,2]].values
        plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='Setosa')
        plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='Versicolor')
        plt.xlabel('Sepal length')
        plt.ylabel('Petal length')
        plt.legend( loc ='upper left')
        plt.show()


    def testPerceptron(self):
        print("\n\n>> Use Perceptron train on first 100 records: eta=0.01 and 10 iterations, then present # of mis classification per iteration")
        print(" ################################################### ")
        df=self.loadIrisData()
        # build training set
        y=df.iloc[0:100,4].values
        y = np.where( y == 'Iris-setosa', -1, 1)
        X=df.iloc[0:100,[0,2]].values
        ppn = Perceptron(0.01,10)
        ppn.fit(X,y)
        plt.plot(range( 1, len( ppn.errors) + 1),ppn.errors, marker ='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of mis-classifications')
        plt.show()


    def testDecisionBoundary(self):
        print("\n\n>> Display the decision boundary to classify an iris in one of the two classes: setosa, versicolor")
        df=self.loadIrisData()
        # build training set with 100 records: the label is in column 5
        y=df.iloc[0:100,4].values
        # transform the classed to numerical value
        y = np.where( y == 'Iris-setosa', -1, 1)
        # take the columns 0,1: sepal and petal length as variables
        X=df.iloc[0:100,[0,2]].values
        ppn = Perceptron(0.01,10)
        ppn.fit(X,y)
        tool.displayDecisionRegions( X, y, classifier = ppn,
        xlabel='sepal length [std]',ylabel='petal length [std]',label0='Setosa',label1='Versicolor')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPerceptron('testLoadIrisData'))
    suite.addTest(TestPerceptron('testPlotIrisData'))
    suite.addTest(TestPerceptron('testPerceptron'))
    suite.addTest(TestPerceptron('testDecisionBoundary'))
    return suite

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLoadIrisData']
    print(" #######################################################")
    print(" Perceptron CLASSIFIER Testing")
    # unittest.main()
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())
    print(" Done...")
    print(" #######################################################")
