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

## Looking at data

The case of the house market data:

```
# compute lot area average by using the pandas
```

## Generic Approach

* Define notebook with the following imports

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
```

Pandas is the primary tool data scientists use for exploring and manipulating data.
Pandas uses DataFrame to hold the type of data you might think of as a table.

See [Kaggle training on pandas](https://www.kaggle.com/learn/pandas).

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
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
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