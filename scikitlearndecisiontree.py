import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import graphviz


#http://scikit-learn.org/stable/modules/tree.html#
filename = 'titanic.csv'
dataset = pd.read_csv("titanic.csv")
new_data = pd.get_dummies(dataset[['Survived','Pclass','Sex','SibSp','Embarked']])
print(new_data.head())
X = new_data.drop('Survived', axis=1)
y = new_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
dot_data = tree.export_graphviz(classifier, out_file=None,
                         feature_names=list(new_data.drop('Survived', axis=1)),
                         class_names="Survival",
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("titanic")
