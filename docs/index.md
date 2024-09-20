# AI and Machine Learning Studies

!!! info "Update"
    Created 2017 - Updated 8/05/2024

Welcome to this repository for machine learning using Python and other cutting-edge technologies! Here, you will find a treasure trove of notes, code samples, and Jupyter notebooks, carefully curated from various reputable sources such as IBM labs, Google, AWS, Kaggle, Coursera, Udemy courses, books, and insightful websites.

This becoming, almost, like a virtual book :smile: 

<figure markdown="span">
  ![Book cover](./diagrams/ai_book_cover.drawio.png)
  <figcaption>A Body of Knowledge about AI driven solutions</figcaption>
</figure>

This web site contents articles, summaries, notebooks and python codes to validate and play with most of the concepts addressed here. 

## Three AI flavors

With the deployment of Generative AI, AI term needs to be more specifics, and all three flavors are useful to address a business problem:

1. **Symbolic AI**: Expert System AI and knowledge graph to represent the human knowledge with rules and relationship semantic. For understanding business rule systems, I recommend reading [Agile Business Rules Development book, from Hafedh Mili and J. Boyer](https://link.springer.com/book/10.1007/978-3-642-19041-4) with [this summary of the methodology](https://jbcodeforce.github.io/methodology/abrd/).
1. **Analytical AI**: the machine learned model from data, analytical algorithms, used to solve analytical tasks such as classification, clustering, predictive scoring, or evaluation. The first neuron-network were used to support better classification on unstructured data like image, and text.
1. **Generative AI**: generate new content (text, image, audio) from human existing large corpus of unstructured data. It uses deep learning, NLP, the transformer architecture, image recognition, voice recognition...

## The AI/ML Market

AI/ML by 2030 will be a $B300 market. Every company is using AI/ML already or consider using it in very short term. 2023 illustrated that AI is part of the world with the arrival of the Generative AI. Some on the main business drivers include:

* Make **faster decisions** by extracting and analyzing data from unstructured documents, voice, video records, transcripts...
* Generate and operationalize **predictive and prescriptive insights** to make decision at the right time.
* Create new content, ideas, conversations, stories, summaries, images, videos or music from question or suggestions (Generative AI).
* Code decision with inference rule to express domain and expert knowledge
* ML helps when developers do not have knowledge of all the rule conditions. Decision trees can be discovered from data.
* Deep learning can adapt to change in the environment and address new scenarios.
* Discover analytic insights within large data set.

The stakeholders interested by AI/ML are CTOs, CPOs, Data Scientists, business analysts who want to derive decision from data and improve their business processes.

## Why Hybrid Cloud?

Enterprises are using data as the main asset to derive empirical decisions and for that, they are adopting big data techniques which means high volume, high variation and high velocity.

In most enterprise data are about customers' behaviors and come from different sources like click stream, shopping cart content, transaction history, historical analytics, IoT sensors,...

Big data need elastic storage and distributed, elastic computing resources. The cloud adoption is really driven by the access to elastic resources, pay as you go, with value-added managed services. 

Private data can stay on-premises servers close to the applications reading the data. Analytics data pre-processing can anonymize data and remove PII data, before uploading to cloud storage. Cloud Storages are used to store large datasets required for training machine learning models, taking advantage of the storage's scalability and performance. So an hybrid cloud strategy is key to support growing adoption of AI and ML solutions.

Model trainings, run as batch processing, for few minutes to few days, and resources can then be released. It is less relevant to buy expensive hardware as CAPEX to do machine learning, when cloud computing can be used.

Most Generative AI, LLMs are deployed as SaaS with API access.

## Data Science major concepts

There are three types of task, data scientists do: 

1. Preparing data to run a model (gathering, cleaning, integrating, transforming, filtering, combining, extracting, shaping...).
2. Running the machine learning model, tuning it and assessing its quality.
3. Communicate the results.

With the adoption of Feature Store technologies, data scientists also prepare the features for reusability and governance.

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


### Big data processing with Map - Reduce

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

[Read my own  Sparck studies](https://jbcodeforce.github.io/spark-studies/) and [Flink](https://jbcodeforce.github.io/flink-studies/).


## Books and other sources

Content is based of the following different sources:

* [Python Machine learning - Sebastian Raschka's book](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130/ref=asc_df_1783555130/?tag=hyprod-20&linkCode=df0&hvadid=312140868236&hvpos=1o7&hvnetw=g&hvrand=12056535591325453294&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9032152&hvtargid=pla-406163981473&psc=1).
* [Collective intelligence - Toby Segaran's book](https://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325/ref=sr_1_2?crid=1UBVCJKMM17Q6&keywords=collective+intelligence&qid=1553021611&s=books&sprefix=collective+inte%2Cstripbooks%2C236&sr=1-2).
* [Stanford Machine learning training - Andrew Ng](https://www.coursera.org/learn/machine-learning).
* [Machine Learning University](https://mlu.corp.amazon.com/course-catalog/)
* [arxiv.org](https://arxiv.org) academic paper on science subjects
* [Dive into deep learning book](https://d2l.ai)
* [Amazon Sagemaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)
* [Kaggle](http://kaggle.com)
* [Papers with code - trends](https://paperswithcode.com/)
* Introduction to Data Sciences - University of Washington.
* [Jeff Heaton - Applications of Deep Neural Networks.](https://github.com/jeffheaton/t81_558_deep_learning)
* [Medium articles on generative AI, ML Analytics...](https://www.medium.com)
* [poe.com](https://poe.com) to search content using LLM
* [Made with ML from Goku Mohandas.](https://madewithml.com/)
* [Vision Transformer github from lucidrains](https://github.com/lucidrains/)