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

* Read csv file and see statistics like min, max, mean, std deviation, and 25,50,75%

```python
home_data = pd.read_csv(a_file_path)
# Print summary statistics
home_data.describe()
```