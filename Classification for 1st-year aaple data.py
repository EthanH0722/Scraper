# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:54:55 2020

@author: hycwy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('aapl_1styear_mined.csv')

#%%

print(df.head())
#%%
data = df.dropna(axis=0, how = 'any')

#%%
print(data.iloc[:,7:17])

#%%
data['per_sum'] = data.iloc[:,7:17].sum(axis=1)

#%%
class_for_10h=[]
for i in data['per_sum']: #per_sum is meaningless for label
    if i >= -10: #threshold
        class_for_10h.append(1)
    else:
        class_for_10h.append(0)

data['class_for_10h'] = pd.Series(class_for_10h)

print(data['class_for_10h'].value_counts())

next_window = data['class_for_10h'][1:].append(pd.Series(1))

#%%
class_for_10th_hour=[]
for i in data['per10']:
    if i >= 0: #threshold
        class_for_10th_hour.append(1)
    else:
        class_for_10th_hour.append(0)

data['class_for_10th_hour'] = pd.Series(class_for_10th_hour)
next_window_10th_hour = data['class_for_10th_hour'][1:].append(pd.Series(1))
#%%
y = next_window_10th_hour
X = data.iloc[:,7:17]  # 7-17 is the index of per1 to per10
X['close'] = data['close']
X['volume'] = data['volume']
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

from sklearn.preprocessing import StandardScaler
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)
#%%
from sklearn.svm import SVC
clf = SVC(C=100,kernel = 'rbf',gamma=0.01).fit(X_train_scaled,y_train)
print('accuracy on training set: %f' % clf.score(X_train_scaled,y_train))
print('accuracy on test set: %f' % clf.score(X_test_scaled,y_test))

#%%
def svc_func(C,gamma):
    svcclf = SVC(C=C,gamma=gamma,kernel='rbf').fit(X_train_scaled,y_train)
    return svcclf.score(X_test_scaled,y_test)
#%%
from bayes_opt import BayesianOptimization
pbounds = {'C':(1,100),'gamma':(0.01,1)}
optimizersvc = BayesianOptimization(
    f=svc_func,
    pbounds=pbounds,
    verbose=2,
    random_state=0,
)
optimizersvc.maximize(n_iter=10,init_points=20)
#%%
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10,random_state=3,bootstrap=True, class_weight=None, criterion='gini',
 max_depth=None, max_features='auto', max_leaf_nodes=None,
 min_samples_leaf=1, min_samples_split=2,
 min_weight_fraction_leaf=0.0, n_jobs=1,
 oob_score=False, verbose=0, warm_start=False).fit(X_train,y_train)

print('accuracy on training set: %f' %forest.score(X_train,y_train))
print('accuracy on test set: %f' %forest.score(X_test,y_test))












