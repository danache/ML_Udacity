# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 09:24:23 2016

@author: danache
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation  import cross_val_score
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import datetime
from IPython.display import display

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_orig = train.copy()
test_orig = test.copy()

train.head()
test.head()
print(train.head())
print(len(train))
print(len(test))
print(len(train.columns))


train = train.drop(labels=["casual", "registered"], axis=1)

train['year'] = train['datetime'].str.extract("^(.{4})")
test['year'] = test['datetime'].str.extract("^(.{4})")

train['month'] = train['datetime'].str.extract("-(.{2})-")
test['month'] = test['datetime'].str.extract("-(.{2})-")

train['day'] = train['datetime'].str.extract("(.{2}) ")
test['day'] = test['datetime'].str.extract("(.{2}) ")

train['time'] = train['datetime'].str.extract(" (.{2})")
test['time'] = test['datetime'].str.extract(" (.{2})")

train[['year', 'month', 'day', 'time']] = train[['year', 'month', 'day', 'time']].astype(int)
test[['year', 'month', 'day', 'time']] = test[['year', 'month', 'day', 'time']].astype(int)

train['dayOfWeek'] = train.apply(lambda x: datetime.date(x['year'], x['month'], x['day']).weekday(), axis=1)
test['dayOfWeek'] = test.apply(lambda x: datetime.date(x['year'], x['month'], x['day']).weekday(), axis=1)

train = train.drop(labels=["datetime"], axis=1)
test = test.drop(labels=["datetime"], axis=1)


train['season'].value_counts()
train = train.join(pd.get_dummies(train.season, prefix='season'))
test = test.join(pd.get_dummies(test.season, prefix='season'))


train = train.drop(labels=["season"], axis=1)
test = test.drop(labels=["season"], axis=1)


train['weather'].value_counts()
train = train.join(pd.get_dummies(train.weather, prefix='weather'))
test = test.join(pd.get_dummies(test.weather, prefix='weather'))


train = train.drop(labels=["weather"], axis=1)
test = test.drop(labels=["weather"], axis=1)

train.corr()

train = train.drop(labels=["atemp"], axis=1)
test = test.drop(labels=["atemp"], axis=1)


target = train['count'].values
train = train.drop(labels=["count"], axis=1)


target = np.log(target)



def bs_fit_and_save(clf, l_train, l_target, l_test, filename):

    clf.fit (l_train, l_target)

    predict_train = clf.predict(l_train)

    print('Variance score: %.2f' % clf.score(l_train, l_target))

    predict_test = clf.predict(l_test)
    predict_test = np.exp(predict_test)

    output = test_orig['datetime']
    output = pd.DataFrame(output)
    predict = pd.DataFrame(predict_test)
    output = output.join(predict)
    output.columns = ['datetime', 'count']
    output.to_csv(filename + ".csv", index=False)
    return clf

def getFile(clf,l_test, filename):

    predict_test = clf.predict(l_test)
    predict_test = np.exp(predict_test)

    output = test_orig['datetime']
    output = pd.DataFrame(output)
    predict = pd.DataFrame(predict_test)
    output = output.join(predict)
    output.columns = ['datetime', 'count']
    output.to_csv(filename + ".csv", index=False)
    return clf

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.cross_validation import ShuffleSplit
from  sklearn.grid_search import GridSearchCV


#clf = RandomForestRegressor(random_state=0, n_estimators=100)
#clf = bs_fit_and_save(clf, train, target, test, "output_RandomForest")
clf = SVR()


cv_sets = ShuffleSplit(train.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)


params = {'C' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'epsilon':[0.2,0.3,0.4,0.5,0.6,0.7,0.8]}

grid = GridSearchCV(clf, params,cv = cv_sets )

grid = grid.fit(train, target)
clf = grid.best_estimator_
clf = getFile(clf,test, "output_SVRCross")
