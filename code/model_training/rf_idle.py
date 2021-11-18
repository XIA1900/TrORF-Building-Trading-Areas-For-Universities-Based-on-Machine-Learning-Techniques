import random
import pymysql as mdb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,cohen_kappa_score,f1_score,recall_score,accuracy_score,precision_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.model_selection import StratifiedKFold,train_test_split,StratifiedShuffleSplit,GridSearchCV,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
np.set_printoptions(suppress=True)


def distribution(df):
    train_size=0.80
    train,test=train_test_split(df,train_size=train_size)
    y_train = train.iloc[:,15:16]
    y_test = test.iloc[:,15:16]
    x_train=train.iloc[:,0:15]
    x_test=test.iloc[:,0:15]
    return x_train,y_train,x_test,y_test

df=pd.read_csv('/users/user/desktop/data.csv')
df['genre']=df['genre'].astype('float32')
df['turnover_normal']=df['turnover_normal'].astype('float32')
df['open_time']=df['open_time'].astype('float32')
df['close_time']=df['close_time'].astype('float32')
df['chain']=df['chain'].astype('float32')
df['year']=df['year'].astype('float32')
df['stu_num']=df['stu_num'].astype('float32')
df['teacher_num']=df['teacher_num'].astype('float32')
df['dep_no']=df['dep_no'].astype('float32')
df['major_no']=df['major_no'].astype('float32')
df['type']=df['type'].astype('float32')
df['master_no']=df['master_no'].astype('float32')
df['phd_no']=df['phd_no'].astype('float32')
df['project_985']=df['project_985'].astype('float32')
df['project_211']=df['project_211'].astype('float32')
df['label']=df['label'].astype('int32')

'''
X=df.iloc[:,0:15]
Y=df.iloc[:,15:16]
Y=Y.values.ravel()
ran=11
for i in range(4,ran):
    KF=KFold(n_splits=i)
    print('i=',i)
    for train_index,test_index in KF.split(df):
        x_train,x_test=X.iloc[train_index],X.iloc[test_index]
        y_train,y_test=Y.iloc[train_index],Y.iloc[test_index]
        clf=RandomForestClassifier(oob_score=True)
        s=clf.fit(x_train,y_train)
        y_pred=clf.predict(x_test)
        print(clf.score(x_test,y_test))
'''

x_train,y_train,x_test,y_test=distribution(df)
y_train=y_train.values.ravel()
y_test=y_test.values.ravel()

clf=RandomForestClassifier(n_estimators=30,oob_score=True)
s=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(clf.score(x_test,y_test))
'''
param_test1={'n_estimators':range(5,71,5)}
gsearch1=GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                       min_samples_leaf=20,
                                                       max_depth=8,
                                                       max_features='sqrt',
                                                       random_state=10),
                         param_grid=param_test1,scoring='precision',cv=5)
gsearch1.fit(x_train,y_train)
print(gsearch1.cv_results_,gsearch1.best_params_,gsearch1.best_score_)

param_test2= {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,
                                 min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10),
   param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(x_train,y_train)
print(gsearch2.cv_results_,gsearch2.best_params_, gsearch2.best_score_)


param_test4= {'max_features':range(3,17,2)}
gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=13, min_samples_split=120,
                                 min_samples_leaf=20 ,oob_score=True, random_state=10),
   param_grid = param_test4,scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(x_train,y_train)
print(gsearch4.best_params_, gsearch4.best_score_)
'''
