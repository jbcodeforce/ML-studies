# Scikit-learn library

Choosing an appropriate classification algorithm for a particular problem task requires practice: each algorithm has its own quirks and is based on certain assumptions.The performance of a classifier, computational power as well as predictive power, depends heavily on the underlying data that are available for learning. 

The five main steps that are involved in training a machine learning algorithm can be summarized as follows:

* Selection of features.
* Choosing a performance metric.
* Choosing a classifier and optimization algorithm.
* Evaluating the performance The sklearn api offers a lot of classifier algorithms and utilities. 

For example the code below loads the predefined IRIS flower dataset, and select the feature 2 and 3, the petals length and width. 

```python
from sklearn import datasets
iris=datasets.load_iris()X=iris.data[:,[2,3]]y=iris.target
```

using numpy unique function to assess the potential classes we got 3 integers representing each class.

```python
print(np.unique(y))
>>[0,1,2]
```

To evaluate how well a trained model performs on unseen data, we will further split the dataset into separate training and test datasets.

```python
from sklearn import cross_validation
# Randomly split X and y arrays into 30% test data and 70% training set 
X_train, X_test, y_train, y_test = cross_validation.train_test_split( X, y, test_size = 0.3, random_state = 0)
```

Many machine learning and optimization algorithms also require feature scaling for optimal performance. `StandardScaler` estimated the parameters mu (sample mean) and delta (standard deviation) for each feature dimension from the training data. The `transform` method helps to standardize the training data using those estimated parameters: mu and delta.  

```python
from sklearn.preprocessing import StandardScaler
# standardize the features for optimal performance of gradient descent
sc=StandardScaler()
# compute mean and std deviation for each feature using fit
sc.fit(X_train)
X_train_std=sc.transform(X_train)
# Note that we used the same scaling parameters to standardize the test set so 
# that both the values in the training and test dataset are comparable to each other.
X_test_std=sc.transform(X_test)
```


Using the training data set, create a Perceptron with 40 iterations and eta = 0.1

```python
from sklearn.linear_model import Perceptron
ppn=Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(X_train_std,y_train)
```

Having trained the model now we can run predictions.

```python
from sklearn.metrics import accuracy_score
y_pred=ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print(' Accuracy: %.2f' % accuracy_score( y_test, y_pred))
```

Scikit-learn also implements a large variety of different performance metrics that are available via the metrics module. For example, we can calculate the classification accuracy of the perceptron on the test set. The Perceptron biggest disadvantage is that it never converges if the classes are not perfectly linearly separable.