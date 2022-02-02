#!/usr/bin/env python3
# coding: utf-8
# Author:   Michael E. Rose <michael.ernst.rose@gmail.com>
"""Solutions for Supervised Machine Learning."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_boston, load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor


# Feature Engineering
# a)
boston = load_boston()
print(boston['DESCR'])
df = pd.DataFrame(boston["data"], columns=boston["feature_names"])

# b)
poly = PolynomialFeatures(degree=2, include_bias=False)
polynomials = poly.fit_transform(df)
print(f"There are {polynomials.shape[1]} polynomials")

# c)
out = pd.DataFrame(polynomials)
out.columns = poly.get_feature_names(df.columns)
out["y"] = boston["target"]
out.to_csv(Path("../output/polynomials.csv"), index=False)


# Regularization
# a)
df = pd.read_csv(Path("../output/polynomials.csv"))

# b)
y = df["y"]
X = df.drop("y", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# b)
linear = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=0.3).fit(X_train, y_train)
lasso = Lasso(alpha=0.3).fit(X_train, y_train)
print(linear.score(X_test, y_test))
print(ridge.score(X_test, y_test))
print(lasso.score(X_test, y_test))

# c)
data = {"linear": linear.coef_, "ridge": ridge.coef_, "lasso": lasso.coef_}
coefs = pd.DataFrame(data, index=df.columns[:-1])
unequal = coefs[(coefs["ridge"] != 0) & (coefs["lasso"] == 0)]
print(f"There are {mask.sum()} coefficients zero for in the Lasso but "
      "non-zero with Ridge")

# d)
fig, ax = plt.subplots(figsize=(10, 30))
coefs.plot.barh(ax=ax)
fig.savefig(Path("../output/polynomials.pdf"))


# Neural Network Regression
# a)
diabetes = load_diabetes()
print(diabetes['DESCR'])
X = diabetes['data']
y = diabetes['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# b)
pipe = Pipeline([("scaler", StandardScaler()),
                 ("nn", MLPRegressor(max_iter=1000, activation="identity",
                                     random_state=42, solver="lbfgs"))])
param_grid = {"nn__hidden_layer_sizes": [(300, 300), (200, 200), (100, 100)],
              "nn__alpha": [0.005, 0.001, 0.0001]}
grid = GridSearchCV(pipe, param_grid, cv=3, return_train_score=True)
grid.fit(X_train, y_train)

# c)
print(grid.best_params_)
print(grid.best_score_)
print(grid.score(X_test, y_test))

# d)
best = grid.best_estimator_
coef_matrices = best._final_estimator.coefs_
df = pd.DataFrame(coef_matrices[0], index=diabetes["feature_names"])
fig, ax = plt.subplots()
sns.heatmap(df, ax=ax)
fig.savefig(Path("./output/nn_diabetes_importances.pdf"))


# Neural Networks Classification
# a)
cancer = load_breast_cancer()
X = cancer['data']
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# c)
pipe = Pipeline([("scaler", MinMaxScaler()),
                 ("nn", MLPClassifier(max_iter=1_000, random_state=42,
                                      solver="lbfgs", activation="tanh"))])
param_grid = {"nn__hidden_layer_sizes": [(20, 10), (20, 20)],
              "nn__alpha": [0.01, 0.001]}
grid = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, scoring="roc_auc")
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
preds = grid.predict(X_test)
print(roc_auc_score(preds, y_test))

# d)
confusion_m = pd.DataFrame(confusion_matrix(y_test, preds))
sns.heatmap(confusion_m, annot=True)
plt.savefig(Path("../output/nn_confusion.pdf"))
