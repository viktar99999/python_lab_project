#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('test.csv')
df.head()
print(df.shape)
print(df.dtypes)
print(df.info())
X = df.iloc[:, 0:19]
y = df.iloc[:, 19]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.88, random_state=42)
standard = StandardScaler()
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)
regressor = RandomForestRegressor(n_estimators=20, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2(y_test, y_pred)
print(r2(y_test, y_pred))
