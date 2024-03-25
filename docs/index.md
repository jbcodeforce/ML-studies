# AI and Machine Learning studies

!!! info "Update"
    Created 2017 - Update 3/25/2024

Welcome to this repository for machine learning using Python and other cutting-edge technologies! Here, you will find a treasure trove of notes, code samples, and Jupyter notebooks, carefully curated from various reputable sources such as IBM labs, AWS, Kaggle, Coursera, Udemy courses, books, and insightful websites.

## The AI/ML Market

AI/ML by 2030 will be a $B300 market. Every company is using AI/ML already or consider using it in very short term. 2023 illustrated that AI is part of the world with the arrival of Generative AI. Some on the main business drivers include:

* Make **faster decisions** by extracting and analyzing data from unstructured documents, voice, video records, transcripts...
* Generate and operationalize **predictive and prescriptive insights** to make decision at the right time.
* Create new content, ideas, conversations, stories, images, videos or music from question or suggestions (Generative AI).

The stakeholders interested by AI/ML are CTOs, CPOs, Data Scientists, business analysts who want to derive decision from data and improve their business processes.

* If we can use a rule based system to address a problem, it is a better choice. It means you can have knowledge of the rules. ML will help when we do not have knowledge of all the rule conditions.
* Deep learning can adapt to change in the environment and address new scenarios.
* Discover insights within large data set.
* Deep learning may not be suitable if we need explanations on what was done by the system, or when the error rate is unacceptable.

## Data Science major concepts

There are three types of task, data scientists do: 

* Preparing data to run a model (gathering, cleaning, integrating, transforming, filtering, combining, extracting, shaping...).
* Running the machine learning model, tuning it and assessing its quality.
* Communicate the results.
* With the new Feature Store technologies, data scientists also prepare the features for reusability and governance.

Enterprises are using data as the main asset to derive empirical decisions and for that, they are adopting big data techniques which means high volume, high variation and high velocity.

In most enterprise data are about customers' behaviors and come from different sources like click stream, shopping cart content, transaction history, historical analytics, IoT sensors,...

### Analytics

The concept of statistical inference is to draw conclusions about a population from sample data using one of the two key methods:

* Hypothesis tests.
* Confidence intervals.

But the truth wears off: previous analysis done on statistical data are less true overtime. Analytics need to be a continuous processing.

#### Hypothesis tests  

The goal of hypothesis test is to compare an experimental group to a control group. There are two types of result:

* **H0** for null hypothesis: this happens when there is no difference between the groups.
* **Ha** for alternative hypothesis: happens when there is statistically significant difference between the groups.

The bigger the number of cases (named study size) the more statistical power we have, and better we are to get better results.

We do not know if the difference in two treatments is not just due to chance. But we can calculate the odds that it is. Which is named the **p-value**.

Statistics does not apply well to large-scale inference problems that big data brings. Big data is giving more spurious results than small data set.

The curse of big data is the fact that when we search for patterns in very, very large data sets with billions or trillions of data points and thousands of metrics,  we are bound to identify coincidences that have no predictive power.


### Map - Reduce

One of the classical approach to run analytics on big data is to use the map-reduce algorithm, which can be summarized as:

* Split the dataset into chunks and process each chunk on a different computer: chunk is typically 64Mb.
* Each chunk is replicated several times on different racks for fault tolerance.
* When processing a huge dataset, the first processing step is to read from distributed file systems and to split data into chunk files.
* Then a record reader reads records from files, then runs the `map` function which is customized for each different problem to solve.
* The combine operation identifies <key, value> with the same key and applies a combine function which should have the associative and commutative properties.
* The output of map function are saved to local storage, then `reduce` task pulls the record per key from the local storage to sort the value and then call the last custom function: reduce

![](./images/map-reduce-1.png){ width=900 }

* System architecture is based on shared nothing, in opposite of sharing file system or sharing memory approach.
* Massive parallelism on thousand of computers where jobs run for many hours. The % of failure of such job is high, so the algorithm should tolerate failure.
* For a given server, a mean time between failure is 1 year then for 10000 servers, we have a likelihood of failure around one failure / hour.
* Distributed FS: very large files TB and PB. Different implementations: Google FS or Hadoop DFS.

Hadoop used to be the map-reduce platform, now [Apache Spark](https://spark.apache.org/) is used for that or [Apache Flink](https://flink.apache.org/).


### What skills needed to grow as data scientist

* [See Udemy blog for what skills to become a data scientist.](https://blog.udemy.com/what-skills-do-you-need-to-become-a-data-scientist/)
* [14 Data Science Projects From Beginner to Advanced Level.](https://blog.udemy.com/data-science-projects)

But the 5 important categories are: 1/ Mathematics, 2/ Statistics, 3/ Python, 4/ 
Data visualization, 5/ Machine learning.

???- question "Explain supervised vs unsupervised"
    Supervised means; build a model from labeled training data. Usupervised we do not know upfront the outcome variable. [read more >> ](./ml/index.md/#supervised-learning)

???- question "Explain the bias-variance tradeoff"
    Bias measures how far off the predictions are from the correct values in general. Variance measures the consistency (or variability) of the model prediction. One way of finding a good bias-variance tradeoff is to tune the complexity of the model via [regularization](./concepts/index.md/#regulaziation). [read more >> ](./concepts/index.md/#variance). Perform [cross-validation]() with hyperparameter tuning.

???- question "Provide examples of regularization techniques"
     The regularization techniques are L1 regularization (Lasso), L2 regularization (Ridge), and elastic net regularization.  [See regularization section](./concepts/index.md/#regulaziation).

???- question "Overfitting / underfitting"

???- question "What are the steps involved in the machine learning pipeline?"

???- question "Compare and contrast classification and regression algorithms"

???- question "What are the evaluation metrics commonly used?"
    Accuracy, Precision  and recall, F1-Score, MSE, Root MSE, R-squared, Area Under the ROC Curve, MAP,... See [this section](./concepts/index.md/#common-performance-metrics-used)

???- question "What is cross-validation, and why is it important in machine learning?"
    Split your data into training and validation sets. Use techniques like k-fold cross-validation to assess the model's performance across different hyperparameter values. Iterate through different regularization strengths and evaluate the model's performance metrics (e.g., accuracy, mean squared error) to find the optimal balance between bias and variance.

???- question "How does feature selection impact machine learning models? Discuss different feature selection methods."

???- question "Explain the concept of ensemble learning."

## Books and other sources

Content is based of the following different sources:

* [Python Machine learning - Sebastian Raschka's book](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130/ref=asc_df_1783555130/?tag=hyprod-20&linkCode=df0&hvadid=312140868236&hvpos=1o7&hvnetw=g&hvrand=12056535591325453294&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9032152&hvtargid=pla-406163981473&psc=1).
* [Collective intelligence - Toby Segaran's book](https://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325/ref=sr_1_2?crid=1UBVCJKMM17Q6&keywords=collective+intelligence&qid=1553021611&s=books&sprefix=collective+inte%2Cstripbooks%2C236&sr=1-2).
* [Stanford Machine learning training - Andrew Ng](https://www.coursera.org/learn/machine-learning).
* [Machine Learning University](https://mlu.corp.amazon.com/course-catalog/)
* [Dive into deep learning book](https://d2l.ai)
* [Amazon Sagemaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)
* [Kaggle](http://kaggle.com)
* [Papers with code - trends](https://paperswithcode.com/)
* Introduction to Data Sciences - University of Washington.
* [Jeff Heaton - Applications of Deep Neural Networks.](https://github.com/jeffheaton/t81_558_deep_learning)
* [poe.com](https://poe.com)
* [Made with ML from Goku Mohandas.](https://madewithml.com/)