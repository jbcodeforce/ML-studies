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

One metric to use is the **Mean Absolute Error**.

```python
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(val_y, val_predictions))
```

A function to get MAE for a decision tree by changing the depth of the tree:

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

# create y and X and then,...

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
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

The goal of feature engineering is simply to make our data better suited to the problem at hand.

We might perform feature engineering to:

* improve a model's predictive performance
* reduce computational or data needs
* improve interpretability of the results

For a feature to be useful, it must have a relationship to the target that your model is able to learn.
Linear models, for instance, are only able to learn linear relationships. 
So, when using a linear model, your goal is to transform the features to make their 
relationship to the target linear.

See [concrete example](https://www.kaggle.com/sinamhd9/concrete-comprehensive-strength).


### Missing values

Remove rows with missing target

```python
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
```

Assess number of missing values in each column

```python
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
```

Avoid dropping a column when there are some missing values, except if the column has a lot of them
and it does not seem to bring much more value.

```python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
```

Use **'imputation'** by assigning the mean value of the column values into the unset cells.
 
```python
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='median')
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```

We can add a boolean column which will have cell set to true when a mean was assigned 
to a missing value. In some cases, this will meaningfully improve results.


```python
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
```

### Categorical variables

A categorical variable takes only a limited number of values. We need to preprocess them to numerical values.


```python
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
# drop categorial variable columns
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```

**Ordinal encoding** assigns each unique value to a different integer.

```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
```
When there are some categorical value in test set that are not in the training set, then 
a solution is to write a custom ordinal encoder to deal with new categories, or drop the column

```python
# first get all categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
```

**One-hot encoding** creates new columns indicating the presence (or absence) of each possible value in the original data.
One-hot encoding does not assume an ordering of the categories

```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to existing numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```

For large datasets with many rows, one-hot encoding can greatly expand the size of the dataset.  
For this reason, we typically will only one-hot encode columns with relatively low cardinality. 
Then, high cardinality columns can either be dropped from the dataset, or we can use ordinal encoding.

```python
# code to get low cardinality
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
```

Remarks that Pandas offers built-in features to do encoding

```python
# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)
```
### Mutual information

The first step is to construct a ranking with a feature utility metric, a function measuring associations between a feature and the target. Then you can choose a smaller set of the most useful features to develop initially.
Mutual information is a lot like correlation in that it measures a relationship between two quantities. 
The advantage of mutual information is that it can detect any kind of relationship, while correlation only detects linear relationships

The least possible mutual information between quantities is 0.0. When MI is zero, the quantities are 
independent: neither can tell you anything about the other.

It's possible for a feature to be very informative when interacting with other features, but not so 
informative all alone. MI can't detect interactions between features. It is a univariate metric.

You may need to transform the feature first to expose the association.

The scikit-learn algorithm for MI treats discrete features differently from continuous features. 
Consequently, we need to tell it which are which. Anything that must have a float dtype is not discrete.
Categoricals (object or categorial dtype) can be treated as discrete by giving them a label encoding.

See example of mutual information in [ml-python/kaggle-training/car-price/PredictCarPrice.py]().

### Discovering new features

* Understand the features. Refer to your dataset's data documentation
* Research the problem domain to acquire domain knowledge: Research yields a variety of formulas for creating potentially useful new features. 
* Study [winning solutions](https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions)
* Use data visualization. Visualization can reveal pathologies in the distribution of a feature or complicated relationships that could be simplified

The more complicated a combination is, the more difficult it will be for a model to learn

Data visualization can suggest transformations, often a "reshaping" of a feature through powers or logarithms

Features describing the presence or absence of something often come in sets. We can aggregate such features by creating a count. 

```python
# creating a feature that describes how many kinds of outdoor areas a dwelling has
X_3 = pd.DataFrame()
X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)
```

Here is an example on how to extract roadway features from the car accidents and compute the number
of such roadway in each accident. 

```python
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
```

Extract Category from a column with string like: `One_Story_1946_and_Newer_All_Styles`

```python
X_4['MSClass'] = df.MSSubClass.str.split('_',n=1,expand=True)[0]
```

**Group transforms** aggregate information across multiple rows grouped by some category. 
With a group transform we can create features like: 

* "the average income of a person's state of residence,"
* "the proportion of movies released on a weekday, by genre."

Using an aggregation function, a group transform combines two features: a categorical feature 
that provides the grouping and another feature whose values we wish to aggregate.
Handy methods include `mean, max, min, median, var, std, count`.

```python
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)
```

If you're using training and validation splits, to preserve their independence, 
it's best to create a grouped feature using only the training set and then join it 
to the validation set. 

```python
# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)
```

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