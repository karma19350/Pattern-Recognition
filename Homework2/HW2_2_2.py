# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:14:21 2019

@author: Qingyang Zhong
"""
#from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score#用pandas加载数据.csv文件，然后用train_test_split分成训练集（75%）和测试集（25%）：
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_curve,auc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.txt',delimiter='\t',header=None)
df.columns=["Sample code number","Clump Thinkness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelisal Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
df.replace(to_replace='?',value=np.nan,inplace=True)

for u in df.columns:
  df[u]=pd.to_numeric(df[u])
#舍去缺省值
df=df.dropna()
new_df=np.array(df)
#划分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(new_df[:,1:10],new_df[:,10],test_size=0.25)
x_train_df=pd.DataFrame(x_train)
y_train_df=pd.DataFrame(y_train)

df_train=pd.concat([x_train_df,y_train_df],axis=1,ignore_index=False)
df_train.columns=["Clump Thinkness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelisal Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
#按照标签分类
group_by_class=df_train.groupby('Class')
train0_df=group_by_class.get_group(0)
train1_df=group_by_class.get_group(1)

x_0=np.array(train0_df)[:,0:9]
x_1=np.array(train1_df)[:,0:9]

m_0=np.mean(x_0,axis=0)#计算每一列的均值
m_1=np.mean(x_1,axis=0)
n_0=x_0.shape[0]#计算每一类的数量
n_1=x_1.shape[0]

cov_0=np.cov(x_0.T)#计算每一类的协方差矩阵
cov_1=np.cov(x_1.T)

Sw= ((n_0-1)*cov_0+ (n_1-1)*cov_1) 
Sw_1=np.linalg.inv(np.mat(Sw))
w=Sw_1*np.mat(m_0-m_1).T*(n_0+n_1)
w_=np.mat(w.T)

p_0=n_0/(n_0+n_1)
p_1=n_1/(n_0+n_1)

predict_train_list=[]
predict_test_list=[]

'''for x in x_train:
  x_=np.mat(x-0.5*(m_0+m_1)).T
  ans=w_*x_-np.log(p_0/p_1)
  print(ans)
  if ans>0:
    flag=0
    predict_train_list.append(flag) 
  else:
    flag=1
    predict_train_list.append(flag)
y_predict=np.array(predict_train_list)
sum=0
scores_list=[]

for j in range(5):
  for i in range(y_predict.shape[0]):
    if y_predict[i]==y_train[i]:
        sum=sum+1
  score=sum/y_predict.shape[0]
  sum=0
  scores_list.append(score)
scores=np.array(scores_list)
print("accuracy",np.mean(scores),scores)'''

for x in x_test:
  x_=np.mat(x-0.5*(m_0+m_1)).T
  ans=w_*x_-np.log(p_0/p_1)
  print(ans)
  if ans>0:
    flag=0
    predict_test_list.append(flag) 
  else:
    flag=1
    predict_test_list.append(flag)
y_predict=np.array(predict_test_list)
sum=0
scores_list=[]

for j in range(5):
  for i in range(y_predict.shape[0]):
    if y_predict[i]==y_test[i]:
        sum=sum+1
  score=sum/y_predict.shape[0]
  sum=0
  scores_list.append(score)
scores=np.array(scores_list)
print("accuracy",np.mean(scores),scores)

confusion_matrix=confusion_matrix(y_test,y_predict)
print (confusion_matrix)
plt.matshow(confusion_matrix)
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.title(u'混淆矩阵')
plt.colorbar()
plt.ylabel(u'实际类型')
plt.xlabel(u'预测类型')
plt.savefig('./average4.png')
plt.show()
