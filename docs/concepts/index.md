# Important ML concepts

## Variance

**Variance** measures the consistency (or variability) of the model prediction for a particular sample instance if we would retrain the model multiple times: if the training set is splitted in multiple subsets, the model can be trained with those subsets and each time the sample instance prediction is run, the variance is computed. If the variability is big, then the model is sensitive to randomness.

## Bias

**Bias** measures how far off the predictions are from the correct values in general. One way of finding a good bias-variance tradeoff is to tune the complexity of the model via [regularization](#regularization). 

## Common Performance Metrics used

During hyperparameter tuning, several performance metrics can be used to evaluate the model's performance. 

* **Accuracy**: measures the proportion of correct predictions over the total number of predictions. It is commonly used for classification problems when the classes are balanced.
* **Precision and Recall**: Precision represents the ratio of true positive predictions to the total predicted positives, indicating the model's ability to correctly identify positive instances. **Recall** measures the ratio of true positive predictions to the total actual positives, indicating the model's ability to find all positive instances. Precision and recall are useful for imbalanced classification problems where the focus is on correctly identifying the positive class.

* **F1 Score:** The F1 score is the harmonic mean of precision and recall. It provides a single metric that balances precision and recall, which is especially useful when both metrics are important.

* **Mean Squared Error (MSE)**: MSE is commonly used as an evaluation metric for regression problems. It measures the average squared difference between the predicted and actual values. Lower MSE values indicate better performance.

* **Root Mean Squared Error (RMSE)**: RMSE is the square root of MSE and provides a more interpretable metric, as it is in the same unit as the target variable. It is widely used in regression tasks.

* **R-squared (R²)**: measures the proportion of the variance in the target variable that can be explained by the model. It ranges from 0 to 1, with higher values indicating better fit. R-squared is commonly used in regression problems.

* **Area Under the ROC Curve (AUC-ROC)**: AUC-ROC is used for binary classification tasks. It measures the model's ability to distinguish between positive and negative instances across different classification thresholds. A higher AUC-ROC score indicates better performance.

* **Mean Average Precision (MAP)**: MAP is often used for evaluating models in information retrieval or recommendation systems. It considers the average precision at different recall levels and provides a single metric to assess the model's ranking or recommendation performance.

## Regularization

**Regularization** is a very useful method to handle collinearity (high correlation among features), filter out noise from data, and eventually prevent overfitting. It adds a penalty term to the regression equation. For regularization to work properly, we need to ensure that all our features are on comparable scales. There are various regularization techniques available, such as L1 regularization (Lasso), [L2 regularization (Ridge)](), and [elastic net regularization](). Each technique adds a regularization term to the model's objective function, which penalizes certain model parameters and helps control their impact on the final predictions. Use regularization strength to determine the degree of influence the regularization term has on the model. It is typically controlled by a hyperparameter (e.g., lambda or alpha) that needs to be tuned.

Higher regularization strength results in simpler models while reduced variance but possibly increased bias.

Lasso regression tends to have a higher bias but lower variance compared to Ridge regression. This means that Lasso might underfit the data if the regularization parameter is too high, but it’s more robust to multicollinearity.

???- info "L1 or Lasso regularization"
    L1 regularization adds a penalty term proportional to the absolute value of the coefficients. It adds the sum of the absolute values of the coefficients multiplied by a tuning parameter (alpha) to the loss function.
    Residual Sum of Squares (RSS), which represents the difference between the predicted and actual values 
    ```python
    # Lasso loss function
    RSS = np.sum((y - X.dot(weights)) ** 2)  # Residual Sum of Squares
    penalty = alpha * np.sum(np.abs(weights))  # L1 penalty term
    return RSS + penalty
    ```
    Due to its ability to eliminate variables, Lasso tends to produce sparse models, where only a subset of features has non-zero coefficients.

???- info "L2 or Ridge regularization"
    It adds a penalty term proportional to the square of the coefficients.
    ```python
    RSS = np.sum((y - X.dot(weights)) ** 2)  # Residual Sum of Squares
    penalty = alpha * np.sum(weights ** 2)  # L2 penalty term
    return RSS + penalty
    ```
    Ridge regression only shrinks the coefficients towards zero but rarely sets them exactly to zero, hence it retains all variables in the model.

See [this demo code tp compare Lasso and Ridge.](https://github.com/jbcodeforce/ML-studies/tree/master/ml-python/demo_lasso_ridge.py)

???- info "elastic net regularization"
    Elastic net regression combines the ridge and lasso penalties to automatically assess relevance of the regularization techniques. The math is (r is the mixing ratio between ridge (r=1) and lasso (r=0) regression):

    ![\Large corr(x,y)](https://latex.codecogs.com/svg.latex?J(\theta)=MSE(\theta)+r(2\alpha \sum_{i=1}^{n} \left| \theta_i \right|) + (1-r)(\frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2 )) 

    Choosing values for alpha and l1_ratio can be challenging; however, the task is made easier through the use of cross validation.

    See [this notebook](https://github.com/jbcodeforce/ML-studies/tree/master/ml-python/kaggle-training/wine_rating/wine_rating_elastic_net_reg.ipynb) to demonstrate Elastic Net Regularization in wine dataset.
    

## Fitting

**Overfitting** is a common problem in machine learning, where a model performs well on training data but does not generalize well to unseen data. If a model suffers from overfitting, we also say that the model has a **high variance**, which can be caused by having too many parameters that lead to a model that is too complex given the underlying data.

### How to deal with overfitting?

A common technique of preventing overfitting is known as regularization.

#### Methods to prevent overfitting

| Method to prevent overfitting | What is it? |
| --- | --- |
| **Simplify the model** | If the current model is overfitting the training data, it may be too complicated. This means it's learning the patterns of the data too well and isn't able to generalize well to unseen data. One way to simplify a model is to reduce the number of layers it uses or to reduce the number of hidden units in each layer. |
| **Use data augmentation** | Data augmentation manipulates the training data in a way so that's harder for the model to learn as it artificially adds more variety to the data. If a model is able to learn patterns in augmented data, the model may be able to generalize better to unseen data. |
| **Use transfer learning**	| Transfer learning involves leveraging the pre-trained weights one model has learned to use as the foundation for the next task. We could use one computer vision model pre-trained on a large variety of images and then tweak it slightly to be more specialized for specific images. |
| **Use dropout layers** | Dropout layers randomly remove connections between hidden layers in neural networks, effectively simplifying a model but also making the remaining connections better. See [torch.nn.Dropout()](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html). |
| **Use learning rate decay** | Slowly decrease the learning rate as a model trains. The closer it gets, the smaller the steps. The same with the learning rate, the closer it gets to convergence, the smaller are weight updates. |
| **Use early stopping** | Early stopping stops model training before it begins to over-fit. The model's loss has stopped decreasing for the past n epochs, we may want to stop the model training here and go with the model weights that had the lowest loss (y epochs prior)|

The decision boundary is the hypothesis that separate clearly the training set.

Decreasing the factor of control of overfitting, C, means the weight coefficients are shrinking so leading to overfitting. Around C=100 the coefficient values stabilize leading to good decision boundaries:

![](./images/iris-boundaries.png){ width=600 }

For C=100 we have now

![](./images/iris-zone-2.png){ width=600 }

### Dealing with under-fitting

Model is under-fitting when it generates poor predictive ability because the model hasn't fully captured the complexity of the training data. To increase model's predictive power, we may look at different techniques:

| Method | Description |
| --- | --- |
| **Add more layers/units to the model** | Model may not have enough capability to learn the required patterns/weights/representations of the data to be predictive. Increase the number of hidden layers/units within those layers.|
| **Tweak the learning rate** | Perhaps the model's learning rate is too high. Trying to update its weights each epoch too much, in turn not learning anything.Try lowering the learning rate. |
| **Use transfer learning**	| Transfer learning may also help preventing under-fitting. It involves using the patterns from a previously working model and adjusting them to our own problem. |
| **Train for longer time** | Train for a more epochs may result in better performance.|
| **Use less regularization** | By preventing over-fitting too much, it may to under-fit. Holding back on regularization techniques can help your model fit the data better.|

Preventing overfitting and under-fitting is still an active area of machine learning research.

