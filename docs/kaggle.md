# Kaggle

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

### Missing values

Remove rows with missing target

```python
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
```

Assess number of missing values in each column

```
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

Use 'imputation' by assigning the mean value of the column values into the unset cells.
 
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

## Categorical variables

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

**XGBoost*,  (extreme gradient boosting) is an implementation of gradient boosting with several additional features focused on performance and speed.

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