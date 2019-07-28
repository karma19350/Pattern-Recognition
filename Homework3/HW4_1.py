# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:01:59 2019

@author: Qingyang Zhong
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

np.random.seed(12)#设置种子
x1=np.random.normal(-2.5,2,350);#负样本
x2=np.random.normal(2.5,1,250);#正样本
label1 = [0 for i in np.arange(0, 350, 1)];
label2 = [1 for i in np.arange(0, 250, 1)];
test_data1 = np.vstack((x1, label1));
test_data2 = np.vstack((x2, label2));
data = np.transpose(np.hstack((test_data1, test_data2)));
x_train, x_test, y_train, y_test = train_test_split(data[:,0],data[:,1],test_size=0.3,random_state=8)

def phi(x, xi):
    phi = 1/(np.sqrt(2*np.pi))*np.e**((-1/2)*pow((x-xi),2))
    return phi

def parzen(xi,x):
  n = len(x)#所有的位置
  y=np.zeros((n,1))
  for i in np.arange(0,n):
    m = 0
    for j in np.arange(0,len(xi)):
        m = m + phi(x[i], xi[j])
    y[i] = m / (len(xi))
  return y

x = np.linspace(-7, 6,num=1000)

x_train_df=pd.DataFrame(x_train)
y_train_df=pd.DataFrame(y_train)

df_train=pd.concat([x_train_df,y_train_df],axis=1,ignore_index=False)
df_train.columns=["Data","Class"]

group_by_class=df_train.groupby('Class')
train0_df=group_by_class.get_group(0)
train1_df=group_by_class.get_group(1)

x_0=np.array(train0_df)[:,0]
x_1=np.array(train1_df)[:,0]

p0=parzen(x_0,x) 
p1=parzen(x_1,x)

plt.plot(x,p0,'r',label='负样本')
plt.plot(x,p1,'b',label='正样本')
plt.title(u"高斯窗非参数估计")
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus'] = False 
plt.legend(loc='lower right')
plt.savefig('./4_1_1.png')

p_0=parzen(x_0,x_test) 
p_1=parzen(x_1,x_test)

def bayesian_classifier(data,rate1,rate2):
    res=np.zeros((len(rate1)))
    right=0.0
    for i in range(0,len(rate1)):
        if rate1[i]>rate2[i]:
            res[i]=0
        else:
            res[i]=1
        if data[i]==res[i]:
            right+=1
    return right/len(rate1)

accuracy1=bayesian_classifier(y_test,p_0,p_1)
print("Accuracy of Bayes Decision for Minimum Errors:",accuracy1)

x_test_df=pd.DataFrame(x_test)
y_test_df=pd.DataFrame(y_test)

df_test=pd.concat([x_test_df,y_test_df],axis=1,ignore_index=False)
df_test.columns=["Data","Class"]

group_by_class=df_test.groupby('Class')
test0_df=group_by_class.get_group(0)
test1_df=group_by_class.get_group(1)

x_test_0=np.array(test0_df)[:,0]
x_test_1=np.array(test1_df)[:,0]

def bayesian_classifier_minimum_risk(data,rate1,rate2,x_test_0,x_test_1):
    res=np.zeros((len(rate1)))
    risk1=np.zeros((len(rate1)))
    risk2=np.zeros((len(rate1)))
    right=0.0
    negative_right=0.0
    positive_right=0.0
    for i in range(0,len(rate1)):
        risk1[i]=10*rate2[i]
        risk2[i]=rate1[i]
        if risk1[i]<risk2[i]:
            res[i]=0
        else:
            res[i]=1
        if data[i]==res[i] and data[i]==0:
            negative_right+=1
            right+=1
        elif data[i]==res[i] and data[i]==1:
            positive_right+=1
            right+=1
    print("Accuracy of Negative Sample:",negative_right/len(x_test_0))    
    print("Accuracy of Positive Sample:",positive_right/len(x_test_1))    
    return right/len(rate1)

accuracy2=bayesian_classifier_minimum_risk(y_test,p_0,p_1,x_test_0,x_test_1)
print("Average Accuracy of Bayes Decision for Minimum Risk:",accuracy2)

