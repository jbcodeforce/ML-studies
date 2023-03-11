# Kaggle and ML tutorial

## Getting started with Titanic

The goal is to find patterns in train.csv that help us predict whether the passengers in test.csv survived.

Example of minimum code for a random forest with 100 decision trees

```python
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# build test and train sets
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```


## Generic Approach

* Define notebook with the following imports

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
```

Pandas is the primary tool data scientists use for exploring and manipulating data.
Pandas uses DataFrame to hold the type of data you might think of as a table which contains an array of individual entries, 
each of which has a certain value. Each entry corresponds to a row (or record) and a column.

See [Kaggle's training on pandas](https://www.kaggle.com/learn/pandas).

* Read csv file and see statistics like min, max, mean, std deviation, and 25,50,75%

```python
home_data = pd.read_csv(a_file_path)
# Print summary statistics
home_data.describe()
# See column names
home_data.columns()
# Select the features using the column names
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
```

* Build the model

```python
# import model from sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)
```

* Make prediction

```python
val_predictions=melbourne_model.predict(val_X)
```

* Validate prediction accuracy

One metric to use is the **Mean Absolute Error** (MAE).

```python
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(val_y, val_predictions))
```

Below is a function to get MAE for a decision tree by changing the depth of the tree:

```python
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

* Persist solution

```python
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```

Then submit to the competition.

## Decision Tree

Use the `DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)` method.

A deep tree with lots of leaves will overfit because each prediction is coming from 
historical data from only the few records at its leaf. But a shallow tree with few leaves 
will perform poorly because it fails to capture as many distinctions in the raw data.

## Random Forest

The random forest uses many trees, and it makes a prediction by averaging the predictions 
of each component tree. It generally has much better predictive accuracy than a single 
decision tree and it works well with default parameters

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# create y and X from input file and then,...

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
predictions = forest_model.predict(val_X)
print(mean_absolute_error(val_y, predictions))
```

Here is an example of running different random forest models:

```python
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
```

## Feature engineering

[See separate note](./features.md)

## Pipelines

To organize pre-processing and model fitting and prediction.

```python
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)
```

## Cross-validation

With cross-validation, we run our modeling process on different subsets of the data to get 
multiple measures of model quality.

For small datasets, where extra computational burden isn't a big deal, we should run cross-validation.
For larger datasets, a single validation set is sufficient. Our code will run faster, and we may have enough data that there's little need to re-use some of it for holdout.

The approach is to use sklearn cross_val_score:

```python
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
```

## Gradient boosting

Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
An **ensemble** combines the predictions of several models.

**XGBoost**,  (extreme gradient boosting) is an implementation of gradient boosting with several additional features focused on performance and speed.

```python
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train,
            early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],)
```

XGBoost has a few parameters that can dramatically affect accuracy and training speed:

* n_estimators: # of model in the ensemble
* early_stopping_rounds: stop iterating when the validation score stops improving
* learning_rate: multiply the predictions from each model by a small number (known as the learning rate) before adding them in.
* n_jobs: equal to the number of cores on your machine, to run in parallel.

## Data Leakage

**Data leakage** (or leakage) happens when your training data contains information 
about the target, but similar data will not be available when the model is used for 
prediction. 
This leads to high performance on the training set (and possibly even the validation data),
 but the model will perform poorly in production.

There are two main types of leakage: target leakage and train-test contamination:

* **Target leakage** occurs when your predictors include data that will not be available 
at the time you make predictions. It is important to think about target leakage in terms 
of the timing or chronological order that data becomes available, not merely whether a 
feature helps make good predictions.
* **train-test contamination**, when we aren't careful to distinguish training data from validation data.
Like running a preprocessing (like fitting an imputer for missing values) before calling `train_test_split()`.

Examples of data leakage:

* to forecast the number of shoelace, every month, the leather used may be a good feature, but
it depends if the value is provided at the beginning of the month as a prediction, or
close to the end of the month as real consumption of the leather used to build the shoes.
* Now if leather represents what the company order to make shoes in the month, then the number
of showlace may be accurate, except if we order them before the leather.
* To predict which patients from a rare surgery are at risk of infection. 
If we take all surgeries by each surgeon and calculate the infection rate among those surgeons.
And then, for each patient in the data, find out who the surgeon was and plug in that surgeon's 
average infection rate as a feature, we will create target leakage if a given patient's outcome 
contributes to the infection rate for his surgeon, which is then plugged back into the 
prediction model for whether that patient becomes infected. We can avoid target leakage 
if we calculate the surgeon's infection rate by using only the surgeries before 
the patient we are predicting for. Calculating this for each surgery in our training 
data may be a little tricky. We also have a train-test contamination problem if we 
calculate this using all surgeries a surgeon performed, including those from the test-set.


## Other Studies

* [Parsing Dates](https://www.kaggle.com/alexisbcook/parsing-dates) from our Data Cleaning course.
* [Geospatial Analysis course](https://www.kaggle.com/learn/geospatial-analysis).
* [Natural Language Processing](https://www.kaggle.com/learn/natural-language-processing)