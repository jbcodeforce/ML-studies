# AI and Machine Learning studies

This repository includes notes, codes to learn how to do machine learning with Python and other technology. This come from studying labs, Udemy, different books and web sites.

## Source for this content

Content is based of the following different sources:

* [Python Machine learning - Sebastian Raschka's book](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130/ref=asc_df_1783555130/?tag=hyprod-20&linkCode=df0&hvadid=312140868236&hvpos=1o7&hvnetw=g&hvrand=12056535591325453294&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9032152&hvtargid=pla-406163981473&psc=1).
* [Collective intelligence - Toby Segaran's book](https://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325/ref=sr_1_2?crid=1UBVCJKMM17Q6&keywords=collective+intelligence&qid=1553021611&s=books&sprefix=collective+inte%2Cstripbooks%2C236&sr=1-2).
* [Stanford Machine learning training - Andrew Ng](https://www.coursera.org/learn/machine-learning).
* Intel ML 101 tutorial.
* [Kaggle](http://kaggle.com)
* Introduction to Data Sciences - University of Washington.
* [AWS SageMaker](https://aws.amazon.com/sagemaker/getting-started/).

## AI/ML Market

AI/ML by 2030 will be a $B300 market. Every company is using AI/Ml already or consider using it in very short term. Some on the main business drivers include:

* Make faster decisions by extracting and analyzing data from documents, records, transcripts...
* Generate and operationalize predictive and prescriptive insights to make decision at the right time.
* Generative AI: create new content, ideas, conversations, stories, images, videos or music from question or suggestions.

The stakeholders interested by AI/ML are CTO, CPO, Data Scientists, business analysts who wants to derive decision from data.

## Data Science major concepts

There are three types of tasks a data scientist do: 

* Preparing data to run a model (gathering, cleaning, integrating, transforming, filtering, combining, extracting, shaping...).
* Running the model, tuning it and assessing its quality.
* Communicate the results.

Enterprises are using data as an important assets to derive empirical decisions and for that they are 
adopting big data which means high volume, high variation and high velocity.

In most enterprise data are about customers and come from different sources like click stream, shopping cart content, historical analytics, sensors,...

### Analytics

The concept of statistical inference is to draw conclusions about a population from sample data using one of the two key methods:

* Hypothesis tests.
* Confidence intervals.

But the truth wears off: previous analysis done on statistical data are less true overtime. Analytics need to be a continuous processing.

#### Hypothesis test  

The goal is to compare an experimental group to a control group. There are two types of result:

  * H0 for null hypothesis: this happens when there is no difference between the groups.
  * Ha for alternative hypothesis: happens when there is statistically significant difference between the groups.

The bigger the number of cases (named study size) the more statistical power you have, and better you are to get better results.

We do not know the difference in two treatments is not just due to chance. But we can calculate the odds that it is. Which is named the **p-value**.

Statistics does not apply well to large-scale inference problem that big data brings. Big data is giving more spurious results than small data set.
The curse of big data is the fact that when you search for patterns in very, very large data sets with billions or trillions 
of data points and thousands of metrics, you are bound to identify coincidences that have no predictive power.

### Map - Reduce

One of the classical approach to run analytics on big data is to use the map-reduce algorithm, which can be summarized as:

- Split the dataset into chunks and process each chunk on a different computer: chunk is typically 64Mb.
- Each chunk is replicated several times on different racks for fault tolerance.
- When processing a huge dataset the first processing step is to read from distributed file system and to split data into chunk files.
- Then a record reader reads records from file, then runs the `map` function which is customized for each different problem to solve.
- The combine operation identifies key, value with the same key and applies a combine function which should have the associative and commutative properties.
- The output of map function are saved to local storage, then `reduce` task pulls the record per key from the local storage to sort the value and then call the last custom function: reduce

 ![](./images/map-reduce-1.png)

- System architecture is based on shared nothing, in opposite of sharing file system or sharing memory approach.
- Massive parallelism on thousand of computers where jobs run for many hours. The % of failure of such job is high, so the algorithm should tolerate failure.
- For a given server a mean time between failure is 1 year then for 10000 servers, we have a likelihood of failure around one failure / hour.
- Distributed FS: very large files TB and PB. Different implementations: Google FS or Hadoop DFS.

Hadoop used to be the map-reduce platform, now [Apache Spark](https://spark.apache.org/) is used for that or [Apache Flink](https://flink.apache.org/).

## Machine Learning

Machine learning is a system that automatically learns programs/ functions from data. There is not programming step. The goal is to find a function to predict **y** from features **Xs**, and continuously measures the prediction performance.

Statistics work on data by applying a model of the world or stochastic models of nature, using linear regression, logistic regression, cox model,... 

Two types of machine learning algorithm, supervised or unsupervised learning.

### Supervised learning

The main goal in supervised learning is to learn a model from labeled training data that allows us to make predictions about unseen or future data. 

**Classification** problem is when we are trying to to predict one of a small number of discrete-valued outputs,
 such as whether it is Sunny (which we might designate as class 0), Cloudy (say class 1) or Rainy (class 2). The class labels are defined as multiple classes or binary classification task, where the machine learning algorithm learns a set of rules in order to distinguish between the possible classes. Classification can be defined when a human assigns a topic label to each document in a corpus, and the algorithm learns how to predict the label. The output is always a set of sets of items. Items could be points in a space or vertices in a graph.

 
Here is an example of data set and classes from the iris flower dataset: 4 features, three potential classes 

```python
feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], 
'target_names': array(['setosa', 'versicolor', 'virginica']
data: array([ 5.1,  3.5,  1.4,  0.2],
            [ 4.9,  3. ,  1.4,  0.2],)

target': array([0, 0, 0, 0,…1,1,1,… 2,2,2])
```

Another subcategory of supervised learning is **regression** classification, where the outcome signal is **continuous value** output. In the table below the Price is the outcome (y) the size # of bedrooms… are features
 
![](./images/house-price.png)

In regression analysis, we are given a number of predictor (explanatory) variables and a continuous response variable (outcome), and we try to find a relationship between those variables that allows us to predict future outcome. 

Three majors components to have for each machine learning classifier:

* **Representation**: define what is the classifier: a rule, a decision tree, a neural network...
* **Evaluation**: how to know if a given classifier is good or bad: how to assess rule result. 
Could be the `# of errors` on some test set, `# of recalls`, squared error, likelihood?... 
We may compute the coverage of a rule: `# of data points` that satisfy the conditions and 
the `accuracy = # of correct predictions / coverage`
* **Optimization**: how to search among all the alternatives, greedy search or gradient descent? 
One idea is to build a set of rules by finding the conditions that maximize accuracy.

When you have a dataset try to humanly inspect the data, and do some plotting diagram with some attribute over other.
Then to select a naive class, look at attribute where you can derive some basic rules. This will build a first hypothesis.
To assess an hypothesis build a **confusion matrix**: a square matrix where column and rows are the different class label of an outcome.
The cells count the number of time the rules classified the dataset. And assess the **accuracy** number: sum good results/ total results.

### Unsupervised learning

Giving a dataset we are able to explore the structure of the data to extract meaningful 
information without the guidance of a known outcome variable or reward function. 

**Clustering** is an exploratory data analysis technique that allows to organize a pile of information into meaningful subgroups (clusters) without having any prior knowledge of their group memberships.

[See deeper dive.](./unsupervised.md)

### Reinforcement learning

The goal is to develop a system (agent) that improves its performance based on interactions with the environment. Through the interaction with the environment, an agent can then uses reinforcement learning to learn a series of actions that maximizes this reward via an exploratory trial-and-error approach or deliberative planning.  

### Unsupervised dimensionality reduction 

This is a commonly used approach in feature preprocessing to remove noise from data, which can also degrade the predictive performance of certain algorithms, and compress the data onto a smaller dimensional subspace while retaining most of the relevant information


### ML System

Building machine learning system includes 4 components as outlined in figure below:

![](./diagrams/ml-steps.drawio.png)

Raw data rarely comes in the form and shape that is necessary for the optimal performance of a learning algorithm. 
Thus, the preprocessing of the data is one of the most crucial steps in any machine learning application. 
Many machine learning algorithms also require that the selected features are on the same scale for optimal performance,
 which is often achieved by transforming the features in the range [0, 1] or a standard normal distribution with zero mean
  and the unit variance.

Some of the selected features may be highly correlated and therefore redundant to a certain degree. 
In those cases, **dimensionality reduction** techniques are useful for compressing the features onto a lower 
dimensional subspace.

Reducing the dimensionality of the feature space has the advantage that less storage space is required, 
and the learning algorithm can run much faster.

To determine whether a machine learning algorithm not only performs well on the training set but also generalizes 
well to new data, we need to **randomly divide** the dataset into a separate **training** and **test** sets.

In practice, it is essential to compare at least a handful of different algorithms in order to train and select 
the best performing model. 

First we have to decide upon a metric to measure performance. One commonly used metric is classification accuracy,
 which is defined as the proportion of correctly classified instances.

After selecting a model that has been fitted on the training dataset, we can use the test dataset to estimate how
 well it performs on this unseen data to estimate the generalization error.

### Model Representation

The notation used:

```sh
m= # of training examples
X= input variables or features
y= output or target
(x(i),y(i)) for the ith training example
```

When the number of features is more than one the problem becomes a linear regression.

Training set is the input to learning algorithm, from which we generate an hypothesis that will be used to map from X to y.

In **regression analysis**, we are given a number of predictor (explanatory) variables and a continuous response variable (outcome),
 and we try to find a relationship between those variables that allows us to predict an outcome. 

Hypothesis function h can be represented as a linear function of x;  

![](https://latex.codecogs.com/svg.latex?h(x)=\sum_{i} \theta_{i} * x_{i}= \theta^{T}*x)

Xo = 1 so a feature is a vector and T is also a row vector of dimension n+1 so H is a matrix multiplication. 
It is called *multivariate linear regression*.

To find the good coefficients ![](https://latex.codecogs.com/svg.latex?\Theta_{i}), the algorithm needs to compare the results H(x) using a cost function:

 ![](https://latex.codecogs.com/svg.latex?J( \theta_0,\theta_1,...,\theta_n) = \frac{1}{2m} \sum_{1}^{m}(h_{\theta} (x_{i}) - y_{i})^2)

The algorithm to minimize the cost function is called the **gradient descent**, and uses the property of the cost function 
being continuous convex linear, so differentiable:

 ![](./images/gradient.png)

The principle is to climb down a hill until a local or global cost minimum is reached. In each algorithm iteration, 
we take a step away from the gradient where the step size is determined by the value of the **learning rate** (alpha) as well as 
the slope of the gradient.

When J(Ti) is already at the local minimum the slope of the tangent is 0 so Tj will not change.
When going closer to the local minimum the slope of the tangent will go slower so the algo will automatically take smaller step.
If alpha is too big, gradient  descent can overshoot the minimum and fail to converge or worse it could diverge.
The derivative is the slope of the tangent at the curve on point Tj. When derivative is close to zero, it means we reach a minima.

When the unit of each feature are very different the gradient descent will take a lot of time to find the minima. 
So it is important to transform each feature so they are in the same scale. (close to: from -1 to 1 range)


### Cost function

One of the key ingredients of supervised machine learning algorithms is to define an 
objective function that is to be optimized during the learning process. This objective 
function is often a cost function that we want to minimize. So the weights update will 
minimize the cost function. The cost function could be the sum squared errors between 
the outcomes and the target label:

![](https://latex.codecogs.com/svg.latex?J(\theta)=\frac{1}{2} * \sum_{i} (y_{i} - \phi (z_{i}))^2)

which translates in python as

```python
errors = (y - output)          
cost = (errors** 2).sum() / 2.0
```

and where 

![](https://latex.codecogs.com/svg.latex?\phi_{i}=\theta^T*X)

in python:

```python
def netInput(self,X):
        # compute z = sum(x(i) * w(i)) for i from 1 to n, add the threshold
        return np.dot(X,self.weights[1:]) + self.weights[0]
```

The cost function is convex continuous linear and can be derived, so that we can use the gradient descent algorithm to find the local minima:

```python
def fit(self,X,y):
    self.weights=np.zeros(1+X.shape[1])
    self.costs=[]
    for _ in range(self.nbOfIteration):
        output = self.netInput(X)
        errors = (y - output)
      # calculate the gradient based on the whole training dataset. Use the matrix * vector 
        self.weights[1:] += self.eta * X.T.dot( errors)
        self.weights[0] += self.eta * errors.sum()
        cost = (errors**2).sum() / 2.0
        self.costs.append(cost)
    return self
```

The weight difference is computed as the negative gradient * the learning rate eta. To compute the gradient of the cost function, we need to compute the partial derivative of the cost function with respect to each weight w(j). 
So putting all together we have:

![](https://latex.codecogs.com/svg.image?\Delta&space;w_{j}=-n\frac{\delta&space;J}{\delta&space;w_{j}}&space;=&space;n&space;\sum_{i}&space;(y_{i}&space;-&space;\phi(z_{i}))x_{i,j}&space;)

the weight update is calculated based on all samples in the training set (instead of updating the weights incrementally after each sample), which is why this approach is also referred to as "batch" gradient descent.
So basically to minimize the cost function we took steps into the opposite direction of a gradient calculated from the entire training set.

