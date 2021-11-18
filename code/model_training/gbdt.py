import random
import pymysql as mdb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,cohen_kappa_score,f1_score,recall_score,accuracy_score,precision_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.model_selection import StratifiedKFold,train_test_split,StratifiedShuffleSplit,GridSearchCV,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.utils import shuffle
np.set_printoptions(precision=5)


def distribution(df):
    train_size=0.80
    train,test=train_test_split(df,train_size=train_size)
    y_train = train.iloc[:,15:16]
    y_test = test.iloc[:,15:16]
    x_train=train.iloc[:,0:15]
    x_test=test.iloc[:,0:15]
    return x_train,y_train,x_test,y_test

df=pd.read_csv('/users/user/desktop/research/final_Data/explotion_1/data.csv')
df=shuffle(df)
feat_labels=df.columns
X=df.iloc[:,0:15]
Y=df.iloc[:,15:16]
Y=Y.values.ravel()
x_train=X[0:138208]
x_verify=X[138208:157952]
x_test=X[157952:199740]
y_train=Y[0:138208]
y_verify=Y[138208:157952]
y_test=Y[157952:199740]
clf = GradientBoostingClassifier(random_state=10)
gbm0 = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
y_pred_ = clf.predict(x_verify)
print(recall_score(y_test, y_pred))
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print("----------")
print(recall_score(y_verify, y_pred_))
print(f1_score(y_verify,y_pred_))
print(accuracy_score(y_verify,y_pred_))
print(precision_score(y_verify,y_pred_))


'''
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(x_train,y_train)
print(gsearch1.best_params_, gsearch1.best_score_)
'''

