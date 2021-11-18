#!/users/use/desktop/research/中间代码/预处理/preprocess.py
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:01:57 2018
从数据库中导出数据
@author: ywei
"""
#类型是target
import random
import pymysql as mdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split    
from sklearn.model_selection import StratifiedShuffleSplit #cross_validation变为了model_selection;在训练集和测试集均匀的分布标签

np.set_printoptions(threshold=np.inf)    #输出所有内容而非省略号

#数据库连接基本设置
conn=mdb.connect(host='127.0.0.1',port=3306,user='root',passwd='980226asdf',db='research',charset='utf8')
conn.autocommit(1) # 如果使用事务引擎，可以设置自动提交事务，或者在每次操作完成后手动提交事务conn.commit()
db_name='research'

try:
    #连接到mysql并转化为pandas
    conn.select_db(db_name)
    sql='select * from store_school'
    df=pd.read_sql(sql,conn)
except:
    import traceback
    traceback.print_exc()
    conn.rollback()
finally:
    conn.close()
    
#将timedelta -> string -> labelcode
def timedelta_process(df):
    df['open_time'].astype('str')
    df['close_time'].astype('str')
    df['open_time']=LabelEncoder().fit_transform(df['open_time'])
    df['close_time']=LabelEncoder().fit_transform(df['close_time'])
    return df
    
#独热编码
def one_hot_encode(df):
    enc=OneHotEncoder()
    enc.fit(df.values)
    l=197440
    arr=np.zeros([197440,112])
    for i in range(l):
        arr[i,:]=enc.transform(df.iloc[1,:].to_frame().T.values).toarray()
      #使用rnn时，不对label编码
    df=pd.DataFrame(arr)
    return df

#连续数据归一化
def minmaxscaler(df):
    l=6
    for i in range(l):
        df.iloc[:,i]=MinMaxScaler().fit_transform(df.iloc[:,i].to_frame())
    return df

#对于dnn和rnn等的数据预处理
'''
df=df[['genre','open_time','close_time','chain','type','project_985','project_211','label','stu_num','teacher_num','master_no','phd_no','dep_no','major_no','turnover_normal','year']]
timedelta_process(df)
df_lisan=df.iloc[:,0:7]
df_label=df.iloc[:,7:8]
df_lianxu=df.iloc[:,8:16]
df_lisan=one_hot_encode(df_lisan)
df_lianxu=minmaxscaler(df_lianxu)
frame=[df_lianxu,df_lisan,df_label]
df=pd.concat(frame,axis=1)
df['stu_num']=df['stu_num'].astype('float32')
df['teacher_num']=df['teacher_num'].astype('float32')
df['dep_no']=df['dep_no'].astype('float32')
df['major_no']=df['major_no'].astype('float32')
df['master_no']=df['master_no'].astype('float32')
df['phd_no']=df['phd_no'].astype('float32')
df['turnover_normal']=df['turnover_normal'].astype('float32')
df['year']=df['year'].astype('float32')
df['label']=df['label'].astype('float32')
for i in range(0,112):
    df[i]=df[i].astype('float32')
df.to_csv('/users/user/desktop/research/final_Data/explotion_1/data_pro.csv',index=False)
'''

#对于随机森林和gbdt等的数据预处理

timedelta_process(df)
df=df[['genre','open_time','close_time','chain','type','project_985','project_211','stu_num','teacher_num','master_no','phd_no','dep_no','major_no','turnover_normal','year','label']]
df.to_csv('/users/user/desktop/research/final_Data/explotion_1/data.csv',index=False)

