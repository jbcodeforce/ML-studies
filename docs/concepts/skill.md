# The data scientist skill set

The following items come from different sources, and grouped in question and answer format. The sources:

* [See Udemy blog for what skills to have to become a data scientist.](https://blog.udemy.com/what-skills-do-you-need-to-become-a-data-scientist/)
* [14 Data Science Projects From Beginner to Advanced Level.](https://blog.udemy.com/data-science-projects)
* LinkedIn blogs and posts

The five important categories are: 1/ Mathematics, 2/ Statistics, 3/ Python, 4/ Data visualization, 5/ Machine learning (including deep learning).

???- question "Explain supervised vs unsupervised"
    Supervised means; build a model from labeled training data. With unsupervised, developers do not know upfront the outcome variable. [read more in this section >> ](../ml/index.md#supervised-learning)

???- question "Explain the bias-variance tradeoff"
    Bias measures how far off the predictions are from the correct values in general. Variance measures the consistency (or variability) of the model prediction. One way of finding a good bias-variance tradeoff is to tune the complexity of the model via [regularization](./index.md/#regularization). [Read more  on variance >> ](./index.md#variance). Perform [cross-validation]() with hyperparameter tuning.

???- question "Provide examples of regularization techniques"
    The regularization techniques are L1 regularization (Lasso), L2 regularization (Ridge), and elastic net regularization.  [See regularization section](./index.md#regularization).

???- question "Overfitting / underfitting"
    Overfitting is a common problem in machine learning, where a model performs well on training data but does not generalize well to unseen data. [Read more in fitting section](./index.md#fitting)

???- question "What are the steps involved in the machine learning pipeline?"
    A typical machine learning pipeline involves several steps, which can be summarized as:

    1. Define the problem
    1. Collect and Prepare Data: Gather, clean and preprocess it to make it suitable for machine learning. This can include tasks such as data wrangling, feature engineering, and data splitting.
    1. Select a Model 
    1. Train the Model: Use the training data to train the model, adjust the model's parameters to minimize the prediction errors.
    1. Evaluate the Model on test data using metrics such as accuracy, precision, recall, and F1 score.
    1. Optimize the Model by tuning hyperparameters, adding or removing features, or trying different models.
    1. Deploy the Model and monitor the model performance to maintain high-quality predictions over time.

???- question "Compare and contrast classification and regression algorithms"
    **Classification** algorithms are used to predict categorical labels or class labels. **Regression** algorithms are used to predict continuous values.

???- question "What are the evaluation metrics commonly used?"
    Accuracy, Precision  and recall, F1-Score, MSE, Root MSE, R-squared, Area Under the ROC Curve, MAP,... See [this section](./index.md/#common-performance-metrics-used)

???- question "What is cross-validation, and why is it important in machine learning?"
    Split your data into training and validation sets. Use techniques like k-fold cross-validation to assess the model's performance across different hyperparameter values. Iterate through different regularization strengths and evaluate the model's performance metrics (e.g., accuracy, mean squared error) to find the optimal balance between bias and variance.

???- question "How does feature selection impact machine learning models? Discuss different feature selection methods."
    Feature selection helps 1/ **reducing overfitting**, which occurs when there is noise in the training data, 2/ improve the **interpretability** of the model by identifying the most important features that contribute to the prediction, 3/ reduce the computational **cost of training** and using the model, 4/ improve the **accuracy** of the model by reducing the noise in the data.

    **Different feature selection methods:**

    * **Filter** methods involve selecting features based on their statistical properties, such as correlation with the target variable or variance. These methods are computationally efficient and can be used as a preprocessing step before training the model.
    * **Wrapper** methods involve selecting features based on their impact on the performance of the model. These methods use a search algorithm to find the subset of features that maximizes the performance of the model. Wrapper methods can be computationally expensive, but they can often lead to better performance than filter methods.
    * **Embedded** methods learn which features are important while training the model. Examples of embedded methods include LASSO (Least Absolute Shrinkage and Selection Operator) and Ridge Regression.
    * **Ensemble** methods involve combining multiple feature selection methods to select a subset of features. These methods can be more robust than individual methods and can lead to better performance. Examples of ensemble methods include recursive feature elimination and random forests.

