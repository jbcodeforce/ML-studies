import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def displayDecisionRegions(X,y,classifier,resolution=0.2,test_idx=None,xlabel=None,ylabel=None,label1='1',label0='0',label2='2',title='classifier'):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap( colors[: len( np.unique( y))])
    # determine the min and max value for the two features used
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # build a grid array, using evenly spaced values within interval min max for each features
    xx1,xx2=np.meshgrid( np.arange( x1_min, x1_max, resolution), np.arange( x2_min, x2_max, resolution))
    # Predict the boundary by using a X = progressing values from x1 min to x1 max (x2 min, x2 max)
    Z = classifier.predict( np.array([ xx1.ravel(), xx2.ravel()]).T)
    # reshape Z as matrix
    Z = Z.reshape( xx1.shape)
    # draw contour plot
    plt.contourf( xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim( xx1.min(), xx1.max())
    plt.ylim( xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate( np.unique( y)):
        if cl == 2:
            aclass=label2
        else:
            if cl == 0 or cl == -1:
                aclass=label0
            else:
                aclass=label1
        plt.scatter( x = X[ y == cl, 0],
                     y = X[ y == cl, 1],
                     alpha = 0.8,
                     c = cmap( idx),
                     marker = markers[ idx],
                     label = aclass)
    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='',alpha=1.0,linewidths=1,marker='o',s=55,label='Test set')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc ='upper left')
    plt.show()
