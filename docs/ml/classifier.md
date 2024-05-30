# Classifiers

**Classification** problem is when we are trying to predict one of a small number of discrete-valued outputs. The class labels are defined as multiple classes or binary classification task, where the machine learning algorithm learns a set of rules in order to distinguish between the possible classes.

Below is a python example of using the iris flower NIST dataset: 4 features, three potential classes:

```python
feature_names= ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 
target_names= ['setosa', 'versicolor', 'virginica']
data= [ 5.1,  3.5,  1.4,  0.2], [ 4.9,  3. ,  1.4,  0.2]

target= [0, 0, 0, 0,…1,1,1,… 2,2,2]
```

Three majors components to have for each machine learning classifier:

* **Representation**: define what is the classifier: a rule, a decision tree, a neural network...
* **Evaluation**: how to know if a given classifier is giving good or bad results: how to asses result rule. 
Could be the `# of errors` on some test set, `# of recalls`, squared error, likelihood?... 
We may compute the coverage of a rule: `# of data points` that satisfy the conditions and the `accuracy = # of correct predictions / coverage`.
* **Optimization**: how to search among all the alternatives, greedy search or gradient descent? 
One idea is to build a set of rules by finding the conditions that maximize accuracy.

For each dataset, try to humanly inspect the data, and do some plotting diagrams with some attributes over others.
Then to select a naive class, look at attribute, where we can derive some basic rules. This will build a first hypothesis.
To assess an hypothesis build a **confusion matrix**: a square matrix where column and rows are the different class label of an outcome. The cells count the number of time the rules classified the dataset. Assess the **accuracy** number: sum good results/ total results.

??? Notes "Code execution"
    All the Classifier Python apps execute well from the python environment in docker. See [environement note.](../coding/index.md/#environments)

## Perceptron

Based on the human neuron model, Frank Rosenblatt proposed an algorithm that would 
automatically learn the optimal weight coefficients that are then multiplied with the input
 features in order to make the decision of whether a neuron fires or not. 
 In the context of supervised learning and classification, such an algorithm could then be
 used to predict if a sample belonged to one class or the other.

The problem is reduced to a binary classification (-1,1), and an activation function 
that takes a linear combination of input X, with corresponding weights vector W, 
to compute the net input as:

```python
z = sum(w(i) * x(i)) i from 1 to n
```

If the value is greater than a threshold the output is 1, -1 otherwise. The function is called `unit step` function. 

If w0 is set to be -threshold and x0=1 then the equation becomes:

![](https://latex.codecogs.com/svg.latex?h(x)=\sum_{i} \theta_{i} * x_{i}= \theta^{T}*x){ width=300 }

The following python functions in a Perceptron class, use numpy library to compute the matrix dot product wT*x:

```python
def netInput(self,X):
     return np.dot(X,self.weights[1:]) + self.weights[0]
   
def predict(self,X):
   return np.where(self.netInput(X)>=0.0,1,-1)
```

The weights are computed using the training set. The value of delta, which is used to update the weight , is calculated by the perceptron learning rule:

![](https://latex.codecogs.com/svg.latex?\Delta(\theta_{j})= \eta*(y_{i} - mean(y_{i}))* x_{i}^j){ width=300 }

eta is the learning rate, Y(i) is the known answer or target for i th sample. The weight update is proportional to the value of X(i)
 
It is important to note that the convergence of the perceptron is only guaranteed if the two classes are linearly separable and the learning rate is sufficiently small.

![](./images/perceptron.png){ width=800 }

The [fit function](https://github.com/jbcodeforce/ML-studies/blob/6073cbb4560386dde09d833878ad0724172ded4b/ml-python/classifiers/Perceptron.py#L24) implements the weights update algorithm.

Test the python Perceptron implementation, uisnf NIST iris dataset. The way to use the perceptron: Create an instance by specifying the eta coefficient and the number of epochs (passes over the training set) to perform

```sh
#under ml-python/classifiers folder
python TestPerceptron.py
```

The test loads the dataset, fit the Perceptron with a training set, plots some sample of the two types of Iris. Then displays the decision boundary to classify an Iris in one of the two classes: setosa, versicolor.

## Adaline

In  ADAptive LInear NEuron classifier, the weights are updated based on a linear activation function (the `Identity` function) rather than a unit step function like in the Perceptron.

![](./images/adaline.png){ width=800 }

```sh
# Start python docker
# under ml-python/classifiers folder
python TestAdaline.py
```

The test works on the Iris dataset too, when we choose a learning rate that is too large, we have an error rate that becomes larger in every epoch because we overshoot the global minimum.

![](./images/ada-learning-rate-1.png)

When the features are standardized (each feature value is reduced by the mean and divided by the standard deviation) the ADALine algorithm converges more quickly.

The two regions illustrates the two classes, with good results: 

![](./images/ada-iris-classes.png)

The following curve shows the cost function results per iteration or epoch

![](./images/adaline-learning-rate.png)


```python
X_std = np.copy(X)
X_std[:,0]=(X[:,0]-np.mean(X[:,0]))/np.std(X[:,0])
X_std[:,1]=(X[:,1]-np.mean(X[:,1]))/np.std(X[:,1])
```
 
The previous approach can take a lot of time when the dataset includes millions of records. A more efficient approach is to take the **stochastic gradient descent** approach. It is used with online training, where the algorithm is trained on-the-fly, while new training set arrives.

The weights are computed with: 

![](https://latex.codecogs.com/svg.latex?w_{i}=\eta*(y_{i}%20-%20\phi%20(z_{i}))*%20x_{i}){ width=300 }

```python
def updateWeights(self,xi,target):
        output = self.netInput( xi)
        error = (target - output)
        self.weights[1:] += self.eta * xi.dot( error)
        self.weights[0] += self.eta * error
        cost = (error** 2)/ 2.0
        return cost
```

To obtain accurate results via stochastic gradient descent, it is important to present it with data in a random order, which is why we want to shuffle the training set for every epoch to prevent cycles.
 
![](./images/ada-iris-boundaries.png)

## Logistic regression

Another classification approach is to use ‘Logistic Regression’ which performs very well on linearly separable set:

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression( C = 1000.0, random_state = 0)
lr.fit( X_train_std, y_train)
lr.predict_proba(X_test_std[0,:])
```

For C=1000 we have the following results:

![](./images/iris-log-reg-1.png){ width=500 }

Logistic regression uses the odds-ratio `P/(1-P)`, P being the probability to have event e: in our case P could be the probability that a set of values for the feature X leads that the sample is part of a class 1. 

In fact, the mathematical model uses the `log (P/(1-P))` as function in the model. It takes input values in the range 0 to 1 and transforms them to values over the entire real number range, which we can use to express a linear relationship between feature values and the log-odds:

![](https://latex.codecogs.com/svg.latex?logit(p(%20y%20=%201%20|%20x))=\sum_{i} \theta_{i} * x_{i}= \theta^{T}*x){ width=400 }

For logistic regression, the hypothesis function is used to predict the probability of having a certain sample X being of class y=1. This is the sigmoid function:

![](https://latex.codecogs.com/svg.latex?\phi(z)=\frac{1}{(1+e^{-z})}){ width=300 }

Here, z is the net input, that is, the linear combination of weights and sample features= W’.x 

The sigmoid function is used as activation function in the classifier:

![](./images/sigmoid-fct-werror.png){ width=600 }

The output of the sigmoid function is then interpreted as the probability of particular sample belonging to class 1 

![](https://latex.codecogs.com/svg.latex?\phi(z)=P(y=1 | x;w)){ width=300 }

given its features X parameterized by the weights W.

Logistic regression can be used to predict the chance that a patient has a particular disease given certain symptoms. As seen before to find the weights W, we need to minimize a cost function, which in the case of logistic regression is:

![](https://latex.codecogs.com/svg.latex?J(w)=C\left [ \sum_{i}^{n} (-y^{i} log(\phi(z^{i})) - (1 - y^{i}))log(1-\phi(z^{i})) \right ] + \frac{1}{2}\left\| w \right\|^2){ width=500 }

The C=1/lambda parameter used in logistic regression api is the factor to control overfitting.

![](https://latex.codecogs.com/svg.latex?\frac{1}{2} ||W||^2) is the regularization bias to penalize extreme parameter weights.


Logistic regression is a useful model for online learning via stochastic gradient descent, but also allows us to predict the probability of a particular event. 



## Maximum margin classification with support vector machines (SVM)

In SVM, the goal is to maximize the margin: the distance between the decision boundary and the training samples.

![](./images/svn-1.png){ width=600 }

The rationale behind having decision boundaries with large margins is that they tend to have a lower generalization error whereas models with small margins are more prone to overfitting.

To prepare the data here is the standard code that is using SciKit `model_selection` to split the input data set into training and test sets and then a standardScaler to normalize values

```python
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=0
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

To train a SVM model using sklearn:

```python
from sklearn.svm import SVC
svm = SVC(kernel='linear',C=1.0,random_state=0)
svm.fit(X_train_std,y_train)
```

See [code in SVM-IRIS.py](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/SVM-IRIS.py)

The SVMs mostly care about the points that are closest to the decision boundary (support vectors).

![](./images/svm-results.png)

The SVM can use Radial Basis Function kernel, to create nonlinear combinations of the original features to project them onto a higher dimensional space via a mapping function phi() where it becomes linearly separable. 

```python
svm = SVC(kernel='rbf',C=10.0,random_state=0, gamma=0.10)
svm.fit(X_train_std,y_train)
```

Gamma is a cut-off parameter for the Gaussian sphere. If we increase the value for gamma, we increase the influence or reach of the training samples, which leads to a softer decision boundary. Gamma at 0.1. Optimizing Gamma is important to avoid overfitting.

## Decision Trees

The decision tree model learns a series of questions to infer the class labels of the samples. 

The algorithm is to start at the tree root and to split the data on the feature that results in the largest information gain (IG). In an iterative process, we can then repeat this splitting procedure at each child node until the leaves are pure. This means that the samples at each node all belong to the same class. In practice, this can result in a very deep tree with many nodes, which can easily lead to overfitting. Thus, we typically want to prune the tree by setting a limit for the maximal depth of the tree.

In order to split the nodes at the most informative features, we need to define an objective function that we want to optimize via the tree learning algorithm. In binary decision trees there are 3 commonly used impurity function: Gini_impurity(), entropy(), and the classification_error().

```python
def gini(p):
    return p *(1-p) + (1-p)*(1-(1-p))

def entropy( p):
    return - p* np.log2( p) - (1 - p)* np.log2(( 1 - p))

def error( p):
    return 1 - np.max([ p, 1 - p])
```

Decision trees are particularly attractive if we care about interpretability. 

See [DecisionTreeIRIS.py code](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/DecisionTreeIRIS.py) and [this DecisionTree notebook](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/DecisionTree.ipynb).

## Combining weak to strong learners via random forests 

Random forests have gained huge popularity in applications of machine learning in 2010s due to their good classification performance, scalability, and ease of use. Intuitively, a random forest can be considered as an ensemble of decision trees. The idea behind ensemble learning is to combine weak learners to build a more robust model, that has a better generalization error and is less susceptible to overfitting.

The only parameter to play with is the number of trees, and the max depth of each tree. The larger the number of trees, the better the performance of the random forest classifier at the expense of an increased computational cost. Scikit-learn provides tools to automatically find the best parameter combinations (via cross-validation)

```python
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = buildTrainingSet()
forest = RandomForestClassifier(criterion ='entropy', n_estimators = 10, random_state = 1, n_jobs = 2)
forest.fit( X_train, y_train)
...
```

the sample size of the bootstrap sample is chosen to be equal to the number of samples in the original training set.

## k-nearest neighbor classifier (KNN)

For KNN, we define some distance metric between the items in our dataset, and find the K closest items.

Machine learning algorithms can be grouped into parametric and nonparametric models. Using parametric models, we estimate parameters from the training dataset to learn a function that can classify new data points without requiring the original training dataset anymore.

With nonparametric models there is no fixed set of parameters, and the number of parameters grows with the training data (decision tree, random forest and kernel SVM). 

KNN belongs to a subcategory of nonparametric models that is described as instance-based learning which are characterized by memorizing the training dataset, and lazy learning is a special case of instance-based learning that is associated with no (zero) cost during the learning process.


The KNN algorithm is fairly straightforward and can be summarized by the following steps:

1. Choose the number of k and a distance metric function.
1. Find the k nearest neighbors of the sample that we want to classify.
1. Assign the class label by majority vote.

In the case of a tie, the scikit-learn implementation of the KNN algorithm will prefer the neighbors with a closer distance to the sample.

See the [KNN notebook](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/KNN.ipynb).

It is important to mention that KNN is very susceptible to overfitting due to the curse of dimensionality. The curse of dimensionality describes the phenomenon where the feature space becomes increasingly sparse for an increasing number of dimensions of a fixed-size training dataset.

The K-nearest neighbor classifier offers lazy learning that allows us to make predictions without any model training but with a more computationally expensive prediction step.

## See also

See also [classifiers done with PyTorch as neural network](../coding/pytorch.md).