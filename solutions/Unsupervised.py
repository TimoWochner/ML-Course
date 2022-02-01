#!/usr/bin/env python3
# coding: utf-8
# Author:   Michael E. Rose <michael.ernst.rose@gmail.com>
"""Solutions for Unsupervised Machine Learning."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import cumsum
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# Principal Component Analysis
# a)
df = pd.read_csv(Path("./data/olympics.csv"), index_col=0)
df.describe()
# Dropping scores is absolutely necessary! It is a combination of all
# the other variables
df = df.drop("score", axis=1)

# b)
scaler = StandardScaler()
X = df.values
X_scaled = scaler.fit_transform(X)
X_scaled.var(axis=0)  # Assert all variables have variance = 1 (includ. some rounding)

# c)
pca = PCA().fit(X_scaled)
out = pd.DataFrame(pca.components_, columns=df.columns)
out.index += 1
out.index.name = "Component"
sns.heatmap(out, cmap="PiYG")
# Interprtation: What do the variables have in common that correlate strongly
# w/ each other (i.e., that jointly load heavily onto components)?
# Sprinting gait; strength; jump

# d)
cumulated = pca.explained_variance_ratio_.cumsum()
THRES = 0.9
n_components = bisect_right(cumulated, THRES) + 1
print(f"You need {n_components} components to explain at least {THRES:.0%} "
      "of the variance.")


# Clustering
# a)
iris = load_iris()
X = iris["data"]

# b)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# c)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=k).fit(X_scaled)
dbscan = DBSCAN(min_samples=2, eps=1).fit(X_scaled)
out = pd.DataFrame({"kmeans": kmeans.labels_,
                    "agglomerative": agg.labels_,
                    "DBSCAN": dbscan.labels_})

# d)
print(f"K-Means: {silhouette_score(X_scaled, kmeans.labels_):.3f}")
print(f"Agglomerative clustering: {silhouette_score(X_scaled, agg.labels_):.3f}")
not_noise = dbscan.labels_ > -1  # Exclude noise otherwise it's treated as cluster
print(f"DBSCAN: {silhouette_score(X_scaled[not_noise], dbscan.labels_[not_noise]):.3f}")

# e)
new = pd.DataFrame(X.T[1:3].T, columns=iris["feature_names"][1:3])
out = pd.concat([out, new], axis=1, sort=True)

# f)
out["DBSCAN"] = out["DBSCAN"].replace(-1, "Noise")

# g)
out = out.melt(id_vars=iris["feature_names"][1:3],
               var_name="Cluster algorithm", value_name="assignment")
sns.catplot(x="sepal width (cm)", y="petal length (cm)",
            col="Cluster algorithm", hue="assignment", data=out)
plt.savefig(Path("./output/cluster_petal.pdf"))
