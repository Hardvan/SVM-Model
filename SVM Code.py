import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Getting Data
iris = sns.load_dataset("iris")

print(iris.head())
print()
print(iris.describe())
print()
print(iris.info())
print()

# EDA
sns.kdeplot(x = "sepal_width", y = "sepal_length",
            data = iris[iris["species"]=="setosa"], shade = True)

sns.pairplot(data = iris, hue = "species", palette = "viridis")

# Training Model
from sklearn.model_selection import train_test_split
X = iris.drop("species", axis = 1)
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 42)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

# Predictions & Evaluation
predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))
print()

# Performing GridSearch
from sklearn.model_selection import GridSearchCV
param_grid = {"C":[0.1,1,10,100,1000,10000],
             "gamma":[1,0.1,0.01,0.001,0.0001,0.00001]}
grid = GridSearchCV(SVC(), param_grid, verbose = 2)
grid.fit(X_train, y_train)

grid_pred = grid.predict(X_test)

print("After Grid Search")
print(confusion_matrix(y_test, grid_pred))
print()
print(classification_report(y_test, grid_pred))
print()



























