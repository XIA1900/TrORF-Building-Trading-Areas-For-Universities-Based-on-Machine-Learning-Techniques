import random
import pymysql as mdb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,cohen_kappa_score,f1_score,recall_score,accuracy_score,precision_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.model_selection import StratifiedKFold,train_test_split,StratifiedShuffleSplit,GridSearchCV,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def distribution(df):
    train_size=0.80
    train,test=train_test_split(df,train_size=train_size)
    y_train = train.iloc[:,121:122]
    y_test = test.iloc[:,121:122]
    x_train=train.iloc[:,0:121]
    x_test=test.iloc[:,0:121]
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

x_train,y_train,x_test,y_test=distribution(df)
y_train=y_train.values.ravel()
y_test=y_test.values.ravel()

clf=SVC()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_train)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))
