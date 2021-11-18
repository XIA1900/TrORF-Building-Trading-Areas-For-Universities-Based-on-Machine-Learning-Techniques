#!/users/use/desktop/research/中间代码/预处理/dnn.py
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:01:57 2018
从数据库中导出数据
@author: ywei
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
warnings.filterwarnings("ignore")

data = pd.read_csv('/users/user/desktop/research/final_Data/explotion_1/data_pro.csv')

def prepareData(data):
    data=shuffle(data)
    match = data[59232:197440]    #70%训练集
    prec = np.array(match.iloc[:,120]).tolist()
    factor = np.array(match.iloc[:,0:120])
    #print('training data loaded')
    
    test = data[19744:59232]      #20%测试集
    prec_t = np.array(test.iloc[:,120]).tolist()
    #match.drop(['121'],axis=1,inplace=True)
    factor_t = np.array(test.iloc[:,0:120])
    #print('testing data loaded')

    verify = data[0:19744]   #10%验证集
    prec_v = np.array(match.iloc[:,120]).tolist()
    factor_v = np.array(match.iloc[:,0:120])
    #print('verifying data loaded')

    X = factor   #输入数据处理
    Y = []
    for m in prec:
        if m==0:
            Y.append([1,0])
        else:
            Y.append([0,1])
    X_t = factor_t
    Y_t = []
    for m in prec_t:
        if m==0:
            Y_t.append([1,0])
        else:
            Y_t.append([0,1])
    X_v = factor_v
    Y_v = []
    for m in prec_v:
        if m==0:
            Y_v.append([1,0])
        else:
            Y_v.append([0,1])

    return X,Y,X_t,Y_t,X_v,Y_v,prec,prec_t,prec_v

#使用l2正则化处理损失
def get_weight(shape, lambd):
    var = tf.Variable(tf.random_normal(shape), dtype= tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    return var

#定义输入层和标签
x = tf.placeholder(tf.float32, shape=(None, 120))
y_ = tf.placeholder(tf.float32, shape=(None, 2))

#定义选用数据大小和网络结构
batch_size = 224
dataset_size = 138208
num_batches = int(dataset_size/batch_size)
layer_dimension = [120,64,32,16,8,4,2]
n_layers = len(layer_dimension)

#定义指数学习率下降
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 3000,0.96,staircase=True)

#初始化各隐藏层节点
cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers-1):
    out_dimension = layer_dimension[i]
    #print(layer_dimension[i])
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.leaky_relu(tf.matmul(cur_layer, weight)+bias)
    in_dimension = layer_dimension[i]

#初始化输出层
out_dimension = layer_dimension[n_layers-1]
print(layer_dimension[n_layers-1])
weight = get_weight([in_dimension, out_dimension], 0.001)
bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
output_layer = tf.nn.leaky_relu(tf.matmul(cur_layer, weight)+bias)


mse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=output_layer))
tf.add_to_collection('losses',mse_loss)

loss = tf.add_n(tf.get_collection('losses'))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss,global_step=global_step)


with tf.Session() as sess:
    #初始化全部节点
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #设定训练次数
    epochs = 10
    #开始训练
    for i in range(epochs):
        print('epoch=',i)
        X,Y,X_t,Y_t,X_v,Y_v,prec,prec_t,prec_v = prepareData(data)    #每一个epoch都是不同的数据
        for j in range(num_batches):
            start = j*batch_size
            end = start+batch_size
            sess.run(train_step, feed_dict={x:X[start:end], y_: Y[start:end]})
            if j % 50==0:
                total_cross_entropy,y_1,y_pred = sess.run([loss,y_,output_layer], feed_dict={x:X, y_: Y})
                print('after %d trainning step(s),cross entrop on training data is %g'% (j, total_cross_entropy))
                y_1 = tf.argmax(y_1, 1).eval()
                y_pred = tf.argmax(y_pred, 1).eval()
                count = 0
                for j in range(len(y_1)):
                    if prec[j] == y_pred[j]:
                        count = count + 1
                print('training set accarcy:',count / len(y_1))
            
                '''
                r_train,r_train_result = sess.run([y_,output_layer], feed_dict={x:X_t, y_:Y_t})
                recall_train_score=recall_score(r_train, r_train_result, average='macro')
                f1_train_score=f1_score(r_train, r_train_result,average='macro')
                precision_train_score=precision_score(r_train,r_train_result,average='macro')
                print('training set recall_score:',recall_train_score)
                print('training set f1_score:',f1_train_score)
                print('training set precision_score:',precision_train_score)
                '''

            
                test_cross_entropy,t_,t_result = sess.run([loss,y_,output_layer], feed_dict={x:X_t, y_:Y_t})
                print('cross entrop on testing data is %g' % (test_cross_entropy))
                t_ = tf.argmax(t_, 1).eval()
                t_result = tf.argmax(t_result, 1).eval()
                count = 0
                for i in range(len(t_)):
                    if prec_t[i]==t_result[i]:
                        count = count+1
                print('testing set accarcy:',count/len(t_))

                '''
                r_test,r_test_result = sess.run([y_,output_layer], feed_dict={x:X_t, y_:Y_t})
                recall_test_score=recall_score(r_test, r_test_result, average='macro')
                f1_test_score=f1_score(r_test, r_test_result, average='macro')
                precision_test_score=precision_score(r_test,r_test_result,average='macro')
                print('testing set recall_score:',recall_test_score)
                print('testing set f1_score:',f1_test_score)
                print('testing set precision_score:',precision_test_score)
                '''
            
            
                verify_cross_entropy,v_,v_result = sess.run([loss,y_,output_layer], feed_dict={x:X_v, y_:Y_v})
                print('cross entrop on verifying data is %g' % (verify_cross_entropy))
                v_ = tf.argmax(v_, 1).eval()
                v_result = tf.argmax(v_result, 1).eval()
                count = 0
                for i in range(len(v_)):
                    if prec_v[i]==v_result[i]:
                        count = count+1
                print('verifying set accarcy:',count/len(v_))

                '''
                r_verify,r_verify_result = sess.run([y_,output_layer], feed_dict={x:X_v, y_:Y_v})
                recall_verify_score=recall_score(r_verify, r_verify_result, average='macro')
                f1_verify_score=f1_score(r_verify, r_verify_result, average='macro')
                precision_verify_score=precision_score(r_verify, r_verify_result,average='macro')
                print('verifying set recall:',recall_verify_score)
                print('verifying set f1_score:',f1_verify_score)
                print('training set precision_score:',precision_verify_score)
                '''
            
                print('----------------------------------------------------')



