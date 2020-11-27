# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
df = pd.read_csv(path)
df.head()
#Code starts here
X = df.drop('Price',axis=1)
y = df['Price']
print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=6)
corr = X_train.corr()
print(corr)


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test,y_pred)
print("r2 Score is {}".format(r2.round(2)))


# --------------
from sklearn.linear_model import Lasso
lasso = Lasso()
# Code starts here
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test,lasso_pred)
print("r2 score for lasso method is {}".format(r2_lasso.round(2)))


# --------------
from sklearn.linear_model import Ridge
ridge = Ridge()
# Code starts here
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test,ridge_pred)
print("r2_score for Ridge Method is {}".format(r2_ridge.round(2)))
# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
regressor = LinearRegression()
score = cross_val_score(regressor, X_train, y_train, cv = 10)
print(score)
mean_score = score.mean()
print(mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
model = make_pipeline(PolynomialFeatures(2),LinearRegression())
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
r2_poly = r2_score(y_test,y_pred)
print(" r2_score of model is {}".format(r2_poly.round(2)))


