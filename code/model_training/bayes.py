#!/users/user/desktop/research/中间代码/模型训练/bayes.py
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:35:45 2019

@author: ywei
"""
import pandas as pd
import numpy as np



data=pd.read_csv('/users/user/desktop/research/final_Data/explotion_1/data.csv')
data_lianxu=data.iloc[:,7:15]
data_lisan=data.iloc[:,0:7]
data_label=data.iloc[:,15]
data_label=data_label.to_frame()   
data_label=data_label.values
data_lianxu=data_lianxu.values

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.X = None
        self.y = None
        # 存储高斯分布的参数(均值, 方差), 因为预测的时候需要, 模型训练的过程中其实就是计算出
        # 所有高斯分布(因为朴素贝叶斯模型假设每个类别的样本集每个特征都服从高斯分布, 固有多个
        # 高斯分布)的参数
        self.parameters = []
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        # 计算每一个类别每个特征的均值和方差
        for i in range(len(self.classes)):
            c = self.classes[i]
            # 选出该类别的数据集
            x_where_c = X[np.where(y == c)]
            # 计算该类别数据集的均值和方差
            self.parameters.append([])
            for j in range(len(x_where_c[0, :])):
                col = x_where_c[:, j]
                parameters = {}
                parameters["mean"] = col.mean()
                parameters["var"] = col.var()
                self.parameters[i].append(parameters)
            return self.parameters

    def calculate_gaussian_probability(self, mean, var, x):
        coeff = (1.0 / (math.sqrt((2.0 * math.pi) * var)))
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
        return coeff * exponent

    # 计算先验概率 
    def calculate_priori_probability(self, c):
        x_where_c = self.X[np.where(self.y == c)]
        n_samples_for_c = x_where_c.shape[0]
        n_samples = self.X.shape[0]
        return n_samples_for_c / n_samples

    # Classify using Bayes Rule, P(Y|X) = P(X|Y)*P(Y)/P(X)
    # P(X|Y) - Probability. Gaussian distribution (given by calculate_probability)
    # P(Y) - Prior (given by calculate_prior)
    # P(X) - Scales the posterior to the range 0 - 1 (ignored)
    # Classify the sample as the class that results in the largest P(Y|X)
    # (posterior)
    
    def classify(self, sample):
        posteriors = []

        # 遍历所有类别
        for i in range(len(self.classes)):
            c = self.classes[i]
            prior = self.calculate_priori_probability(c)
            posterior = np.log(prior)

            # probability = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xN|Y)
            # 遍历所有特征 
            for j, params in enumerate(self.parameters[i]):
                # 取出第i个类别第j个特征的均值和方差
                mean = params["mean"]
                var = params["var"]
                # 取出预测样本的第j个特征
                sample_feature = sample[j]
                # 按照高斯分布的密度函数计算密度值
                prob = self.calculate_gaussian_probability(mean, var, sample_feature)
                # 朴素贝叶斯模型假设特征之间条件独立，即P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y), 
                # 并且用取对数的方法将累乘转成累加的形式
                posterior += np.log(prob)

            posteriors.append(posterior)

        # 对概率进行排序
        index_of_max = np.argmax(posteriors)
        max_value = posteriors[index_of_max]
        return posterior
        #return self.classes[index_of_max]

    # 对数据集进行类别预测
    def predict_lianxu(self, X):
        y_pred = []
        for sample in X:
            y = self.classify(sample)
            y_pred.append(y)
        return np.array(y_pred)

    
    

clf = NaiveBayes()
para=clf.fit(data_lianxu,data_label)
print(para)
posr=np.array(clf.predict_lianxu(data_lianxu))
print(posr)
