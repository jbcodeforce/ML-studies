# Mathematical foundations

## Covariance 

![\Large cov(x,y)=\sum_{i}^{} (x_{i} - u_{x})(y_{i} - u_{y})](https://latex.codecogs.com/svg.latex?cov(x,y)=\sum_{i}^{} (x_{i} - u_{x})(y_{i} - u_{y})) 

## Correlation 

![\Large corr(x,y)](https://latex.codecogs.com/svg.latex?corr(x,y)=\frac{cov(x,y)}{\sqrt {\sum_{i}^{} (x_{i} - u_{x})^2} * \sqrt {\sum_{i}^{} (y_{i} - u_{y})^2 }) 

## Bayesian

Used to compute the probability of a given outcome given a data set:

Bayes theorem:  

![](https://latex.codecogs.com/svg.latex?P(A|B) = P(B|A) P(A) / P(B))

* incorporate prior knowledge.
* opposite is frequentist approach using data and find decision

 ![](./images/bayen.png)

## Data distributions

[See this notebook presenting](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/Distributions.ipynb) python code on different data distributions like Uniform, Gaussian, Poisson.

## Normalization

Normalization of ratings means adjusting values measured on different scales to a notionally common scale, often prior to averaging. 
In statistics, normalization refers to the creation of shifted and scaled versions of statistics,
where the intention is that these normalized values allow the comparison of corresponding normalized values for different
 datasets in a way that eliminates the effects of certain gross influences, as in an anomaly time series.

Feature scaling used to bring all values into the range [0,1]. This is also called unity-based normalization.

 ![](https://latex.codecogs.com/svg.latex?X'=(X-Xmin)/(Xmax-Xmin))