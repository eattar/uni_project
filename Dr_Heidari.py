# python 3.7.3
# sklearn 0.20.3

import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_excel(os.path.realpath(r'Book1.xlsx'))
df.dropna(how='any', inplace=True)

X = df.iloc[0:, 1:]
y = df.iloc[0:, 0]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=65)
poly = PolynomialFeatures(4)
X_train_transform = poly.fit_transform(X_train)
X_test_transform = poly.fit_transform(X_test)
lr = LinearRegression().fit(X_train_transform, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train_transform, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test_transform, y_test)))
