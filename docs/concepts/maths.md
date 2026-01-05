# Mathematical foundations

## Covariance

![\Large cov(x,y)=\sum_{i}^{} (x_{i} - u_{x})(y_{i} - u_{y})](https://latex.codecogs.com/svg.latex?cov(x,y)=\sum_{i}^{} (x_{i} - u_{x})(y_{i} - u_{y}))

## Correlation

![\Large corr(x,y)](https://latex.codecogs.com/svg.latex?corr(x,y)=\frac{cov(x,y)}{\sqrt {\sum_{i}^{} (x_{i} - u_{x})^2} * \sqrt {\sum_{i}^{} (y_{i} - u_{y})^2 }) 

## Probability

* the probability of two independent events happening at the same time: 

![](https://latex.codecogs.com/svg.latex?P(A \wedge B) = P(A) * P(B))

* the probability of two dependent events happening at the same time:

![](https://latex.codecogs.com/svg.latex?P(A|B) = P(B|A) * P(A))

* the probability of disjoint events A and B (they are mutually exclusive) is:

![](https://latex.codecogs.com/svg.latex?P(A \vee B) = P(A) + P(B))

* if A and B are not mutually exclusive:

![](https://latex.codecogs.com/svg.latex?P(A \vee B) = P(A) + P(B) - P(A \wedge B))

*What is the probability that a card chosen from a standard deck will be a Jack or a heart?* becomes P(Jack or Heart) = P(Jack) + P(heart) - P( jack of Hearts) = 16/52

## Bayesian

In machine learning, there are two main approaches: the **Bayesian** approach and the **frequentist** approach. The Bayesian approach is based on probability theory and uses Bayes' theorem to update probabilities based on new data. The frequentist approach, on the other hand, is based on statistical inference and uses methods such as hypothesis testing and confidence intervals to make decisions.

### Bayes theorem

Bayes' theorem provides a way to update probabilities based on new evidence. Understanding it involves three levels:

1. **Knowing the formula** - being able to plug in numbers
2. **Understanding why it's true** - grasping the derivation
3. **Recognizing when to apply it** - identifying real-world situations

#### The formula

The Bayes formula for the probability of hypothesis H given evidence E:

![](https://latex.codecogs.com/svg.latex?P(H|E)=\frac{P(E|H)\cdot%20P(H)}{P(E)})

Where:

* **P(H)** - the **prior**: probability of the hypothesis before seeing evidence
* **P(E|H)** - the **likelihood**: probability of the evidence assuming the hypothesis is true
* **P(E)** - the **evidence**: total probability of seeing the evidence under all hypotheses

    ![](https://latex.codecogs.com/svg.latex?P(E)=P(H)\cdot%20P(E|H)+P(not H)\cdot%20P(E|not H))

* **P(H|E)** - the **posterior**: updated probability of the hypothesis after seeing evidence

Another view of this formula:

![](./images/bayen.png){ width=500 }

#### The Steve example

Consider Steve, described as "meek and tidy soul, with a need for order and a passion for detail." Is Steve more likely a librarian or a farmer?

The intuitive answer might be librarian, but Bayes' theorem requires considering:

* **Prior probability**: There are roughly 20 farmers for every librarian in the population -> P(H) = 1 /( 1 + 20) = 1/21
* **Likelihood**: What fraction of librarians vs farmers fit Steve's description?

Even if 40% of librarians (4 librarians for 10 of them), this is P(E|H), and only 10% of farmers match the character description (20 of the 200 farmers), the prior matters:

* Librarians matching: 1 × 0.40 = 0.40
* Farmers matching: 20 × 0.10 = 2.00

Steve is about 5x more likely to be a farmer than a librarian.

The P(Librarian given description) = 4 /(4 + 20) = 16.7%

* Rationality is not about knowing facts, it's about recognizing which facts are relevant.
* Seeing evidence restricts the space of possibilities

#### Visual representation

A geometric interpretation uses a unit square representing the sample space:

![](./images/bayes.drawio.png)

1. Divide the square into regions representing each hypothesis (proportional to prior probabilities)
2. Within each region, shade the area where the evidence holds (proportional to likelihood)
3. The posterior is the ratio of shaded hypothesis area to total shaded area

This visualization shows how restricting to cases where evidence holds (conditioning) changes the probability.

#### Key takeaways

* Always consider base rates (priors) before updating beliefs
* New evidence updates but does not replace prior knowledge
* Context and representative sampling affect the validity of conclusions

The Bayesian approach handles uncertainty and complex data well. Frequentist methods are more common when data is abundant and variable relationships are well-defined.

[See the conditional probability notebook](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/ConditionalProbabilityExercise.ipynb) exercise to simulate the probability of buying thing knowing the age and previous buying data: `totals` contains the total number of people in each age group and `purchases` contains the total number of things purchased by people in each age group.

[See this video from 3Blue1Brown](https://www.youtube.com/watch?v=HZGCoVF3YvM) for a geometric interpretation of Bayes' theorem.

## Data distributions

[See this notebook presenting](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/Distributions.ipynb) some python code on different data distributions like Uniform, Gaussian, Poisson. It can be executed in VScode using the pytorch kernel.

## Normalization

Normalization of ratings means adjusting values measured on different scales to a notionally common scale, often prior to averaging.

In statistics, normalization refers to the creation of shifted and scaled versions of statistics,
where the intention is that these normalized values allow the comparison of corresponding normalized values for different
 datasets in a way that eliminates the effects of certain gross influences, as in an anomaly time series.

Feature scaling used to bring all values into the range [0,1]. This is also called unity-based normalization.

![](https://latex.codecogs.com/svg.latex?X'=(X-Xmin)/(Xmax-Xmin))

## Sigmoid function

[The Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) has a S shaped curve, one of them being the logistic function, to change a real to a value between 0 and 1.

![](https://latex.codecogs.com/svg.latex?\phi(z)=\frac{1}{(1+e^{-z})}){ width=200 }

It is used as an activation function of artificial neuron. The logistic sigmoid function is invertible, and its inverse is the [logit function](https://en.wikipedia.org/wiki/Logit):

![](https://latex.codecogs.com/svg.latex?logit(p)=\log_{e}{\frac{p}{1-p}}){ width=200 }

P being a probability, ![](https://latex.codecogs.com/svg.latex?{\frac{p}{1-p}}) is the corresponding odds.