'''
Created on Jul 4, 2016

@author: jeromeboyer
'''
import unittest

from matplotlib.colors import ListedColormap

from Adaline import AdalineGradiantDescent as ad
from Adaline import AdalineStockaticGradiantDescent as asgd
import Tool as tool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TestAdaline(unittest.TestCase):

    def loadIrisData(self):
        print(" ... load IRIS data")
        df=pd.read_csv('../../data/iris_data.txt',header=None)
        return df

    def prepareFeatures(self):
        df=self.loadIrisData()
        y=df.iloc[0:100,4].values
        y = np.where( y == 'Iris-setosa', -1, 1)
        X=df.iloc[0:100,[0,2]].values
        return X,y

    def prepareTraining(self):
        X,y=self.prepareFeatures()
        X_std = np.copy(X)
        X_std[:,0]=(X[:,0]-np.mean(X[:,0]))/np.std(X[:,0])
        X_std[:,1]=(X[:,1]-np.mean(X[:,1]))/np.std(X[:,1])
        return X_std,y


    def testAdelineGD(self):
        print("\n\n>> Test standardized Adaline with two values for eta")
        X,y=self.prepareFeatures()
        fig, ax = plt.subplots( nrows = 1, ncols = 2, figsize =( 8, 4))
        # the eta is too high, the errors are becoming larger as local minima is overshoot
        ada1 = ad( nbOfIteration = 10, eta = 0.01)
        ada1.fit( X, y)
        ax[0].plot(range( 1, len( ada1.costs) + 1), np.log10( ada1.costs), marker ='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log( Sum-squared-error)')
        ax[0].set_title('Adaline - Learning rate 0.01')
        ada2 = ad( nbOfIteration = 10, eta = 0.0001)
        ada2.fit( X, y)
        print(ada2.costs)
        ax[1].plot(range( 1, len( ada2.costs) + 1), np.log10( ada2.costs), marker ='x')
        ax[1].set_xlabel(' Epochs')
        ax[1].set_ylabel('log( Sum-squared-error)')
        ax[1].set_title('Adaline - Learning rate 0.0001')

        print("\nThe left chart shows what could happen if we choose a learning rate"
        + "that is too large:\n Instead of minimizing the cost function, the error becomes\n"
        + "larger in every epoch because we overshoot the global minimum"
        + "\n\n Close the window to continue the test")
        plt.show()

    def testStandardizedFeatures(self):
        print(">> Test standardized Gradient Descent with more iterations")
        X_std,y=self.prepareTraining();
        ada1 = ad( nbOfIteration = 15, eta = 0.01)
        ada1.fit(X_std,y)
        print("\n The two regions illustrates the two classes, with good results")
        tool.displayDecisionRegions( X_std, y, classifier = ada1,xlabel='sepal length [std]',ylabel='petal length [std]')
        print("\n The curve shows the cost function results per iteration or epoch")
        plt.plot(range( 1, len( ada1.costs) + 1), np.log10( ada1.costs), marker ='o')
        plt.xlabel(' Epochs')
        plt.ylabel('log( Sum-squared-error)')
        plt.title('Adaline - Learning rate 0.01')
        plt.show()


    def testStochasticGD(self):
        print("\n>> Test stochastic Gradient Descent")
        X_std,y=self.prepareTraining();
        ada=asgd(nbOfIteration = 15, eta = 0.01,randomState=1)
        ada.fit(X_std, y)
        print(">> Display Decision regions")
        tool.displayDecisionRegions( X_std, y, classifier = ada,
           xlabel='sepal length [std]',ylabel='petal length [std]')

        print(">> Display cost function, showing better results")
        plt.plot(range( 1, len( ada.costs) + 1),  ada.costs, marker ='o')
        plt.xlabel(' Epochs')
        plt.ylabel('Average costs')
        plt.title('Adaline - Learning rate 0.01')
        plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    print(" #######################################################")
    print(" ADELINE CLASSIFIER Testing")

    unittest.main()
    print(" Done...")
    print(" #######################################################")
