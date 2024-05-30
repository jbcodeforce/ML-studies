# Unsupervised  Learning

Unsupervised algorithms don't make use of a target. The goals is to learn some property of the data, to represent the structure of the features in a certain way.

Clustering is the technic to group data based  on how similar they  are to each other.
Adding a feature of **cluster labels** can help machine learning models untangle complicated relationships of space or proximity.
Cluster feature  is categorical.

The motivating idea for adding cluster labels is that the clusters will break up complicated 
relationships across features into simpler chunks. Our model can then just learn the simpler chunks 
one-by-one instead having to learn the complicated whole all at once. It's a "divide and conquer" strategy.

K-means clustering measures similarity using euclidean distance. It creates clusters by placing a 
number of points, called **centroids**, inside the feature-space. Each point in the dataset is 
assigned to the cluster of whichever centroid it's closest to. The "k" controls how many centroids  to create.

k-means clustering is sensitive to scale, so it is a good idea rescale or normalize data with extreme values.
As a rule of thumb, if the features are already directly comparable (like a test result at different
 times), then you would not want to rescale. On the other hand, features that aren't on comparable 
 scales (like height and weight) will usually benefit from rescaling.

For example  in housing price prediction, lot area and living area  may  need  to be scaled to avoid big lot to impact
too much the price.

Here is an example of cluster labels:

```python
# Define a list of the features to be used for the clustering
features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF","GrLivArea"]

# Standardize
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)


# Fit the KMeans model to X_scaled and create the cluster labels
kmeans = KMeans(n_clusters=10,n_init=10 random_state=0)
X["Cluster"] =  kmeans.fit_predict(X_scaled)
```

Use cluster distance:

```python
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)

# Create the cluster-distance features using `fit_transform`
X_cd = kmeans.fit_transform(X_scaled)

# Label features and join to dataset
X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
X = X.join(X_cd)
```

