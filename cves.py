#!/usr/bin/env python
# coding: utf-8


# Initialisation
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pandas.io.json import json_normalize

import ast


# Import dataset
df = pd.read_json("circl-cve-search-expanded.json", lines=True)


# Option to read from the csv but parses incorrect formats for some of the JSON objects
# df = pd.read_csv("cve-search.csv")

# Export as csv for later use (may cause problems with incorrect formats)
# df.to_csv('cve-search.csv', index=False)


# check import
# print(df.columns)


# Trim data to select columns
# Published date, CVE id, CVSS score
df2 = df[['Published', 'id', 'cvss', 'access', 'impact', 'capec', 'cwe']]


dfa = df2[df2.cwe != 'Unknown']
dfb = dfa[dfa.capec.notnull()]
dfc = dfb[dfb.astype(str)['capec'] != '[]']

dfd = dfc[dfc.astype(str)['access'] != '{}']
dfe = dfd[dfd.impact != '{}']

print(dfe)


# Extract and convert capec to columns
capecs = []

for i in range(0,len(dfe['capec'])):
    codes = json_normalize(dfe['capec'].iloc[i])
    codes_list = codes['id'].values.tolist()
    capecs.append(codes_list)
    
cap = pd.Series(capecs)

mlb = preprocessing.MultiLabelBinarizer()

cap_enc = pd.DataFrame(mlb.fit_transform(cap),
                   columns=mlb.classes_,
                   index=cap.index)

cap_enc = cap_enc.add_prefix("capec_")

print(cap_enc)


# Convert access / impact to columns
# One hot encoding on the access / impact columns

acs = pd.DataFrame(dfe['access'].values.tolist())
acs.columns = 'access.' + acs.columns
acs = acs.dropna()

ipt = pd.DataFrame(dfe['impact'].values.tolist())
ipt.columns = 'impact.' + ipt.columns
ipt = ipt.dropna()

acs_enc = pd.concat([pd.get_dummies(acs['access.authentication'], prefix='acs_auth'),
                     pd.get_dummies(acs['access.complexity'], prefix='acs_comp'),
                     pd.get_dummies(acs['access.vector'], prefix='acs_vect')
                    ],axis=1)

ipt_enc = pd.concat([pd.get_dummies(ipt['impact.confidentiality'], prefix='ipt_conf'),
                     pd.get_dummies(ipt['impact.availability'], prefix='ipt_avail'),
                     pd.get_dummies(ipt['impact.integrity'], prefix='ipt_intg')
                    ],axis=1)

cwe_enc = pd.get_dummies(dfe['cwe'])

# Get dfe without modified columns
col = dfe.columns.difference(['id','Published','access','impact', 'cwe', 'capec'])

cves = pd.concat([dfe[col], acs_enc, ipt_enc],join='inner',axis=1)

print(cves)


y = cves.columns.difference(['cvss'])
cves_X = cves[y]
cves_y = cves['cvss']

cves_X_train, cves_X_test, cves_y_train, cves_y_test = train_test_split(cves_X, cves_y)

print(cves_X_train.shape)

print(cves_X.corr())


# Linear Regression
# adapted from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(cves_X_train, cves_y_train)

# Make predictions using the testing set
cves_y_pred = regr.predict(cves_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print('Mean squared error:           %.2f'
      % mean_squared_error(cves_y_test, cves_y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(cves_y_test, cves_y_pred))

# Plot outputs
plt.scatter(cves_X_test.iloc[:,0], cves_y_test,  color='black')
plt.plot(cves_X_test.iloc[:,0], cves_y_pred, color='blue', linewidth=0.5)

plt.xticks(())
plt.yticks(())

plt.show()


# Ridge regression
reg = linear_model.Ridge(alpha=.5)

# Train the model using the training sets
reg.fit(cves_X_train, cves_y_train)

# Make predictions using the testing set
cves_y_pred_ridge = reg.predict(cves_X_test)

# The coefficients
print('Coefficients: \n', reg.coef_)

# The mean squared error
print('Mean squared error:           %.2f'
      % mean_squared_error(cves_y_test, cves_y_pred_ridge))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(cves_y_test, cves_y_pred_ridge))

# Plot outputs
plt.scatter(cves_X_test.iloc[:,0], cves_y_test,  color='black')
plt.plot(cves_X_test.iloc[:,0], cves_y_pred_ridge, color='blue', linewidth=0.5)

plt.xticks(())
plt.yticks(())

plt.show()

