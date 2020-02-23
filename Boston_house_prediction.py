# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 04:27:39 2020

@author: Arpit
"""
## importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# importing dataset
from sklearn.datasets import load_boston

dataset = load_boston()
# loading all features
X = dataset.data
# loading target values
y = dataset.target
# getting all the feature name
dataset.feature_names

# feature scalling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
#sc_y=StandardScaler()
X=sc_X.fit_transform(X)


# splliting dataset into training set test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,
                                                 random_state=0)

# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# predicting result on test_set
y_pred_lin = lin_reg.predict(X_test)

# predicting accuracy of linear model
lin_reg_score= lin_reg.score(X_test,y_test)*100
# 63.54% accuracy

## Using backword elimination to throw useless feature 
## it will improve algorithm performance

import statsmodels.api as sm
# adding column of 1's
X=np.append(arr = np.ones((len(X),1)),values=X,axis=1)

## backword elimination
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
reg_OLS = sm.OLS(endog = y, exog=X_opt).fit()
reg_OLS.summary()
# x3 and x7 least important so remove
X_opt = X[:,[0,1,2,4,5,6,8,9,10,11,12,13]]
reg_OLS = sm.OLS(endog = y, exog=X_opt).fit()
reg_OLS.summary()

# optimized dataset
X_train_o,X_test_o,y_train_o,y_test_o = train_test_split(X_opt,y,test_size=0.25,
                                                 random_state=0)
# optimized model without 2 features
lin_reg_o=LinearRegression()
lin_reg_o.fit(X_train_o,y_train_o)

# predicting results on test set 
y_pred_o=lin_reg_o.predict(X_test_o)

# score of multiple linear regression 
multi_reg_score = lin_reg_o.score(X_test_o,y_test_o)*100
# 63.69% accuracy


## Support vector regression'
from sklearn.svm import SVR
sv_reg = SVR(kernel='rbf')

sv_reg.fit(X_train,y_train)

#predictinng results on test set
y_pred_svr=sv_reg.predict(X_test)

# score of svr
svr_score = sv_reg.score(X_test,y_test)*100
# 50.91%

