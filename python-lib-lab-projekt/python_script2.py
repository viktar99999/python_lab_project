#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import sklearn
file = input("Введите название csv файла: ")
df = pd.read_csv(file)
df.head()
print(df.shape)
print(df.dtypes)
print(df.info())
print(df['Price'])
print(df['Price'].min())
print(df['Price'].max())
print(df['Price'].mean())
print(df['Price'].median())
