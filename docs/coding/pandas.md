# [Pandas](https://pandas.pydata.org/)

Pandas is the primary tool data scientists use for exploring and manipulating data.

Pandas uses `DataFrame` to hold the type of data a table which contains an array of individual entries, each of which has a certain value. Each entry corresponds to a row (or record) and a column.

Pandas supports the integration with many file formats or data sources out of the box (csv, excel, sql, json, parquet,…)

```python
import pandas as pd
masses_data=pd.read_csv('mammographic_masses.data.txt',na_values=['?'],names= ['BI-RADS','age','shape','margin','density','severity'])
masses_data.head()
```

## How to

### Work on the data

```python
masses_data.describe()
# Search for row with no data in one of the column
masses_data.loc[masses_data['age'].isnull()]
# remove such rows
masses_data.dropna(inplace=True)
```

### Transform data for Sklearn


```python
all_features = masses_data[['age', 'shape',
                             'margin', 'density']].values


all_classes = masses_data['severity'].values

feature_names = ['age', 'shape', 'margin', 'density']
```

## Sources

* [Pandas getting started](https://pandas.pydata.org/docs/getting_started/index.html)
* [Kaggle's training on pandas](https://www.kaggle.com/learn/pandas).
