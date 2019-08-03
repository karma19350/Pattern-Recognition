# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:54:44 2019

@author: Qingyang Zhong
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cross_validation import cross_val_score
#读取数据
x_df = pd.read_csv('./feature_selection_X.txt',delimiter='\t',header=None)
y_df = pd.read_csv('./feature_selection_Y.txt',header=None)
for u in x_df.columns:
  x_df[u]=pd.to_numeric(x_df[u])
for u in y_df.columns:
  y_df[u]=pd.to_numeric(y_df[u])

x_data=np.array(x_df)
y_data=np.array(y_df)
print(y_data.shape)
print(x_data.shape)

from sklearn.model_selection import train_test_split
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_data, y_data, test_size=0.3, random_state=3)
y_train_new=y_train_new.ravel()
y_test_new=y_test_new.ravel()
print(y_test_new.shape)

#支持向量机方法
from sklearn.svm import SVC   
clf = SVC(kernel='poly', degree=3,gamma=10,verbose=1)
clf.fit(x_train_new, y_train_new) 
y_predition_train=clf.predict(x_train_new)
y_predition_test=clf.predict(x_test_new)

total=0
right=0
for i in range(len(y_predition_train)):
    if y_predition_train[i]==y_train_new[i]:
        right+=1
    total+=1
acc=float(right/total)  
print('SVM train accuarcy: ' + str(acc))
total=0
right=0
for i in range(len(y_predition_test)):
    if y_predition_test [i]==y_test_new[i]:
        right+=1
    total+=1
acc=float(right/total)   
print('SVM test accuarcy: ' + str(acc))
  
#Logistic Regression      
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix

classifier=LogisticRegression()
classifier.fit(x_train_new,y_train_new)
y_predict=classifier.predict(x_test_new)

total=0
right=0
for i in range(len(y_predict)):
    if y_predict[i]==y_test_new[i]:
        right+=1
    total+=1
acc=float(right/total)  
print('Logistic Regression train accuarcy: ' + str(acc))

confusion_matrix=confusion_matrix(y_test_new,y_predict)
plt.matshow(confusion_matrix,cmap=plt.cm.Blues)
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.title(u'混淆矩阵')
plt.colorbar()
plt.ylabel(u'实际类型')
plt.xlabel(u'预测类型')
for first_index in range(2):
    for second_index in range(2):
        plt.text(first_index, second_index, confusion_matrix[second_index][first_index], va='center', ha='center')
plt.savefig('./average.png')
plt.show()

x_train_df=pd.DataFrame(x_train_new)
y_train_df=pd.DataFrame(y_train_new)
y_train_df.columns=[1000]
df_train=pd.concat([x_train_df,y_train_df],axis=1,ignore_index=False)

#按照标签分类
print(df_train.columns)
group_by_class=df_train.groupby(1000)
train0_df=group_by_class.get_group(0)
train1_df=group_by_class.get_group(1)
print(np.array(train0_df).shape)
print(np.array(train1_df).shape)
x_0=np.array(train0_df)[:,0:1000]
x_1=np.array(train1_df)[:,0:1000]

Jw=[]
def takeOrder(elem):
    return elem[1]

for i in range(1000):
    m_0=np.mean(x_0[:,i],axis=0)#计算每一列的均值
    m_1=np.mean(x_1[:,i],axis=0)
    n_0=x_0[:,i].shape[0]#计算每一类的数量
    n_1=x_1[:,i].shape[0]
    S_0=0
    S_1=0
    for j in range(n_0):
        S_0=S_0+pow((x_0[j,i]- m_0),2)
    for k in range(n_1):
        S_1=S_1+pow((x_1[k,i]- m_1),2)
    Jw.append([i+1,pow((m_0- m_1),2)/(S_0 + S_1)])
Jw.sort(key=takeOrder,reverse = True)
Jw=np.array(Jw)
print(Jw[0:10,0])
plt.figure(num=1, figsize=(8, 5)) 
plt.plot(Jw[:,1],'*-')
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.ylabel('Jw')
plt.savefig('./fisher.png')
plt.show()

Corr=[]
for i in range(1000):
    cor=np.corrcoef(x_train_new[:,i].T,y_train_new)
    Corr.append([i+1,abs(cor[0][1])])

Corr.sort(key=takeOrder,reverse = True)
Corr=np.array(Corr)
print(Corr[0:10,0])
plt.figure(num=2, figsize=(8, 5)) 
plt.plot(Corr[:,1],'*-')
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.ylabel('相关系数')
plt.savefig('./cor.png')
plt.show()
