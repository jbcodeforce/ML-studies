import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

plt.style.use("seaborn-whitegrid")


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


df = pd.read_csv("./Automobile_data.csv")
print(df.head())

X = df.copy()
y = X.pop("price")


# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]

mi_scores = make_mi_scores(X, y, discrete_features)
print(mi_scores[::3])
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
sns.relplot(x="curb_weight", y="price", data=df);
