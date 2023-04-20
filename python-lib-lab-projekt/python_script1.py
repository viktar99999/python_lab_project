#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as r2
df = pd.read_csv('test.csv')
df.head()
df.shape
df.dtypes
df.info()
X = df.iloc[:, 0:19]
y = df.iloc[:, 19]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.88, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
regressor = RandomForestRegressor(n_estimators=20, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2(y_test, y_pred)
print(r2(y_test, y_pred))
