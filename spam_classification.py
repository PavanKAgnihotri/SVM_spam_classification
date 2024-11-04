#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:14:57 2024

@author: pavan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', header=None)
#print(df)

x = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels
#print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

def train_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

#Logistic Regression
lr = LogisticRegression(random_state=42)
lr_accuracy = train_model(lr, x_train, y_train, x_test, y_test)
print(f'Logistic Regression\nAccuracy: {lr_accuracy*100:.3f}')

#Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt_accuracy = train_model(dt, x_train, y_train, x_test, y_test)
print(f'Decision Tree\nAccuracy: {dt_accuracy*100:.3f}')

#Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb_accuracy = train_model(gb, x_train, y_train, x_test, y_test)
print(f'Gradient Boosting\nAccuracy: {gb_accuracy*100:.3f}')

print('--------------------')

print('Using PCA')
k = 10
pca = PCA(n_components=k)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

#Training models with PCA data
#Logistic Regression
lr_pca_accuracy = train_model(lr, x_train_pca, y_train, x_test_pca, y_test)
#Decision Tree
dt_pca_accuracy = train_model(dt, x_train_pca, y_train, x_test_pca, y_test)
#Gradient Boosting
gb_pca_accuracy = train_model(gb, x_train_pca, y_train, x_test_pca, y_test)
print(f'Logistic Regression\nAccuracy: {lr_pca_accuracy*100:.3f}')
print(f'Decision Tree\nAccuracy: {dt_pca_accuracy*100:.3f}')
print(f'Gradient Boosting\nAccuracy: {gb_pca_accuracy*100:.3f}')
print('--------------------')

# PCA visualization
x_pca = PCA(n_components=2).fit_transform(x)
plt.figure(figsize=(8,6))
plt.scatter(x_pca[y==0,0], x_pca[y==0,1], label='Non-Spam', alpha=0.5, marker='o')
plt.scatter(x_pca[y==1,0], x_pca[y==1,1], label='Spam', alpha=0.5, marker='x')
plt.legend()
plt.title('Principal Component Analysis')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.savefig('PCA.png')
plt.show()
plt.clf()

# tSNE visualization
X_tsne = TSNE(n_components = 2, learning_rate = 'auto', init = 'random', perplexity = 30).fit_transform(x_pca)
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label='Non-Spam', alpha=0.5, marker='o')
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label='Spam', alpha=0.5, marker='x')
plt.legend()
plt.title('t-SNE visualization')
plt.xlabel('tSNE 1')
plt.ylabel('tSNE 2')
plt.savefig('tSNE.png')
plt.show()

print('Done')