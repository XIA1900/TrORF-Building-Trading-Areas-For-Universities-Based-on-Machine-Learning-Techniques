#!/users/user/desktop/research/中间代码/模型训练/randomForest.py
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 00:35:45 2019

@author: ywei
"""

import random
import pymysql as mdb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,cohen_kappa_score,recall_score,accuracy_score,precision_score,f1_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.model_selection import StratifiedKFold,train_test_split,StratifiedShuffleSplit,GridSearchCV,KFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

np.set_printoptions(suppress=True)

df=pd.read_csv('/users/user/desktop/research/final_Data/explotion_1/data.csv')
df=shuffle(df)
X=df.iloc[:,0:15]
Y=df.iloc[:,15:16]
Y=Y.values.ravel()
x_train=X[0:138208]
x_verify=X[138208:157952]
x_test=X[157952:199740]
y_train=Y[0:138208]
y_verify=Y[138208:157952]
y_test=Y[157952:199740]
KF=KFold(n_splits=10)
clf=RandomForestClassifier(n_estimators=70,max_depth=13,min_samples_split=70,max_features=9,oob_score=True)
scores=cross_val_score(clf,x_train,y_train,cv=10)
print(scores)

'''
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

clf=RandomForestClassifier(n_estimators=70,max_depth=13,min_samples_split=70,max_features=9,oob_score=True)
s=clf.fit(x_train,y_train)
y_pred_test=clf.predict(x_test)
y_pred_verify=clf.predict(x_verify)
importances=clf.feature_importances_
features=clf.n_features_
'''
'''
print("testing")
print(accuracy_score(y_test,y_pred_test))
print(precision_score(y_test,y_pred_test))
print(f1_score(y_test,y_pred_test))
print(recall_score(y_test,y_pred_test))
print("prediction")
print(accuracy_score(y_verify,y_pred_verify))
print(precision_score(y_verify,y_pred_verify))
print(f1_score(y_verify,y_pred_verify))
print(recall_score(y_verify,y_pred_verify))
'''

'''
plt.title('Feature Importance of RF')
plt.bar(range(x_train.shape[1],importances[indices],color='grey',align='center'))
plt.xticks(range(x_train.shape[1]),feat_labels,rotation=90)
plt.xlim([-1,x_train.shape[1]])
plt.tight_layout()
plt.show()
'''

'''
acc=[]
pre=[]
rec=[]
ran=11
x_=[4,5,6,7,8,9,10]
for i in range(4,ran):
    df=shuffle(df)
    X=df.iloc[:,0:15]
    Y=df.iloc[:,15:16]
    Y=pd.DataFrame(Y.values.ravel())
    KF=KFold(n_splits=i)
    print('i=',i)
    for train_index,test_index in KF.split(df):
        x_train,x_test=X.iloc[train_index],X.iloc[test_index]
        y_train,y_test=Y.iloc[train_index],Y.iloc[test_index]
        clf=RandomForestClassifier(oob_score=True)
        s=clf.fit(x_train,y_train)
        y_pred=clf.predict(x_test)
        acc.append(accuracy_score(y_test,y_pred))
        pre.append(precision_score(y_test,y_pred))
        rec.append(recall_score(y_test,y_pred))

y1=np.array(acc)
y2=np.array(pre)
y3=np.array(rec)
l1=plt.plot(x_,y1,'r--',label='accuracy')
l2=plt.plot(x_,y2,'g--',label='precision')
l3=plt.plot(x_,y3,'b--',label='recall')
plt.figure(figsize=(9,6))
plt.title('K-Fold Cross Validation Results of RF')
plt.plot(x_,y1,'ro-',x_,y2,'g+-',x_,y3,'b^-')
plt.ylabel('Results')
plt.xlabel('K')
plt.tight_layout()
plt.show()
'''

'''#70
param_test1={'n_estimators':range(5,71,5)}
gsearch1=GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                       min_samples_leaf=20,
                                                       max_depth=8,
                                                       max_features='sqrt',
                                                       random_state=10),
                         param_grid=param_test1,scoring='precision',cv=5)
gsearch1.fit(x_train,y_train)
print(gsearch1.best_params_,gsearch1.best_score_)
'''

'''
#70,13,70
param_test2= {'n_estimators':range(5,71,5),'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20),'max_features':range(3,16,2)}
gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70,
                                 min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10),
   param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(x_train,y_train)
print(gsearch2.cv_results_,gsearch2.best_params_, gsearch2.best_score_)
'''

'''
param_test4= {'max_features':range(3,16,2)}
param_test4= {'max_features':range(3,16,2)}
gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=13, min_samples_split=120,
                                 min_samples_leaf=20 ,oob_score=True, random_state=10),
   param_grid = param_test4,scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(x_train,y_train)
print(gsearch4.best_params_, gsearch4.best_score_)
'''
