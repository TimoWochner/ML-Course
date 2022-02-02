#!/usr/bin/env python3
# coding: utf-8
# Author:   Michael E. Rose <michael.ernst.rose@gmail.com>
"""Solutions for pandas and seaborn exercises."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Tips
# a)
tips = sns.load_dataset("tips")

# b)
replace = {"Thur": "Thursday", "Sun": "Sunday", "Sat": "Saturday",
           "Fri": "Friday"}
tips["day"] = tips["day"].replace(replace)

# c)
g = sns.relplot(x="tip", y="total_bill", data=tips, markers=True, style="day",
                hue="day", col="sex")
plt.savefig("../output/tips.pdf")


# Occupations
# a)
FNAME = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"
df = pd.read_csv(FNAME, sep="|")
df = df.set_index("user_id")

# b)
print(df.tail(10))
print(df.head(25))

# c)
print(df.info())

# d)
occ_counts = df["occupation"].value_counts()
print(type(occ_counts))

# e)
print(occ_counts.shape[0])  # or print(occ_counts.nunique())
print(occ_counts[0])

# f)
occ_counts = occ_counts.sort_index()
fig, ax = plt.subplots()
occ_counts.plot.bar(ax=ax)
ax.set(xlabel="Occupation")
fig.savefig("../output/occupations.pdf")


# Iris
# a)
FNAME = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ['sepal length (in cm)', 'sepal width (in cm)', 'petal length (in cm)',
        'petal width (in cm)', 'class']
iris = pd.read_csv(FNAME, header=None, names=cols)

# b)
iris.loc[10:30, 'petal length (in cm)'] = None

# c)
iris = iris.fillna(1.0)

# d)
iris.to_csv('./output/iris.csv', index=False, sep=",")

# d)
cont = iris.select_dtypes("float")
fig = sns.catplot(data=cont)
fig.set_xticklabels(rotation=45)
plt.savefig('../output/iris.pdf')


# Memory
# a)
df = pd.read_csv("https://query.data.world/s/wsjbxdqhw6z6izgdxijv5p2lfqh7gx")

# b)
print(df.info())
print(df.info(memory_usage="deep"))

# c)
df_copy = df.copy().select_dtypes(include=[object])

# d)
df_copy.describe()

# e)
# Yes absolutely! It saves a lot of memory

# f)
CUTOFF = 0.49*df.shape[0]
few_unique = [col for col in df_copy.columns if df_copy[col].nunique() <= CUTOFF]
for col in few_unique:
    df[col] = df[col].astype('category')

# g)
# c)
df_copy = df.copy().select_dtypes(include=[object])

# d)
df_copy.describe()

# e)
# Yes absolutely! It saves a lot of memory

# f)
CUTOFF = 0.49*df.shape[0]
few_unique = [col for col in df_copy.columns if df_copy[col].nunique() <= CUTOFF]
for col in few_unique:
    df[col] = df[col].astype('category')

# g)
print(df.info(memory_usage="deep"))

# h)
# Yes, by specifying dtypes of columns using the dtype parameter

# e)
df["v_line_score"] = df["v_line_score"].astype(str)  # Temporary fix because of https://issues.apache.org/jira/browse/ARROW-14087
df.to_csv("../output/large_file.csv")
df.to_feather("../output/large_file.ftr")
# The reduction amounts to about 40%, because feather stores data in a columnar format
