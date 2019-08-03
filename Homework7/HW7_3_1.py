# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:48:02 2019

@author: Qingyang Zhong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#from sklearn.cross_validation import cross_val_score
#读取数据
x_df = pd.read_csv('./feature_selection_X.txt',delimiter='\t',header=None)
y_df = pd.read_csv('./feature_selection_Y.txt',header=None)
for u in x_df.columns:
  x_df[u]=pd.to_numeric(x_df[u])
for u in y_df.columns:
  y_df[u]=pd.to_numeric(y_df[u])

x_data=np.array(x_df)
y_data=np.array(y_df)
data=np.array(pd.concat([x_df,y_df],axis=1,ignore_index=False))
y_data=y_data.ravel()


def takeOrder(elem):
    return elem[1]

def FisherSelection(data):
    df_train=pd.DataFrame(data)
    #按照标签分类
    #print(df_train.columns)
    group_by_class=df_train.groupby(1000)
    train0_df=group_by_class.get_group(0)
    train1_df=group_by_class.get_group(1)
    x_0=np.array(train0_df)[:,0:1000]
    x_1=np.array(train1_df)[:,0:1000]

    Jw=[]
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
    featureList=Jw[0:10,0].astype(int)-1
    print("Fisher feature:",featureList)
    return featureList

def CorrSelection(data):
    x_data=data[:,:-1]
    y_data=data[:,-1]
    Corr=[]
    for i in range(1000):
        cor=np.corrcoef(x_data[:,i].T,y_data)
        Corr.append([i+1,abs(cor[0][1])])
    Corr.sort(key=takeOrder,reverse = True)
    Corr=np.array(Corr)
    featureList=Corr[0:10,0].astype(int)-1
    print("correlation coefficient feature:",featureList)
    return featureList

from sklearn.model_selection import KFold
def main():
  kf = KFold(n_splits=10, shuffle=True)
  SVM_fisher_list=[]
  Logistic_fisher_list=[]
  SVM_corr_list=[]
  Logistic_corr_list=[]
  for train_index, test_index in kf.split(data):
      print("*************************************************")
      train=data[train_index]
      test=data[test_index]
      
      featureList = FisherSelection(train)
      train_new = train[:,[featureList[0],featureList[1],featureList[2],featureList[3],featureList[4],featureList[5],featureList[6],featureList[7],featureList[8],featureList[9],-1]]
      test_new = test[:,[featureList[0],featureList[1],featureList[2],featureList[3],featureList[4],featureList[5],featureList[6],featureList[7],featureList[8],featureList[9],-1]]
      SVM_fisher_list.append(SVM_Classify(train_new,test_new))
      Logistic_fisher_list.append(Logistic_Regression(train_new,test_new))
      
      featureList = CorrSelection(train)
      train_new = train[:,[featureList[0],featureList[1],featureList[2],featureList[3],featureList[4],featureList[5],featureList[6],featureList[7],featureList[8],featureList[9],-1]]
      test_new = test[:,[featureList[0],featureList[1],featureList[2],featureList[3],featureList[4],featureList[5],featureList[6],featureList[7],featureList[8],featureList[9],-1]]
      SVM_corr_list.append(SVM_Classify(train_new,test_new))
      Logistic_corr_list.append(Logistic_Regression(train_new,test_new))
  print("*************************************************")
  print("Fisher + SVM Classifier:",np.mean(SVM_fisher_list))
  print("Fisher + Logistic_Regression:",np.mean(Logistic_fisher_list))
  print("Correlation coefficient + SVM Classifier:",np.mean(SVM_corr_list))
  print("Correlation coefficient + Logistic_Regression:",np.mean(Logistic_corr_list))
 
#支持向量机方法
      
from sklearn.svm import SVC  
from sklearn.linear_model.logistic import LogisticRegression

 
def SVM_Classify(train,test):
    #clf = SVC(kernel='poly', degree=3,gamma=10,verbose=1)
    clf = SVC(kernel='linear',verbose=1)
    x_train=train[:,:-1]
    y_train=train[:,-1]
    x_test=test[:,:-1]
    y_test=test[:,-1]
    clf.fit(x_train, y_train) 
    y_predition_test=clf.predict(x_test)
    total=0
    right=0
    for i in range(len(y_predition_test)):
        if y_predition_test [i]==y_test[i]:
            right+=1
        total+=1
    acc=float(right/total)   
    print('SVM test accuarcy: ' + str(acc))
    return acc
  
#Logistic Regression方法    
def Logistic_Regression(train,test):
    x_train=train[:,:-1]
    y_train=train[:,-1]
    x_test=test[:,:-1]
    y_test=test[:,-1]
    classifier=LogisticRegression()
    classifier.fit(x_train,y_train)
    y_predict=classifier.predict(x_test)
    total=0
    right=0
    for i in range(len(y_predict)):
        if y_predict[i]==y_test[i]:
            right+=1
        total+=1
    acc=float(right/total)  
    print('Logistic Regression train accuarcy: ' + str(acc))
    return acc


if __name__=='__main__':
    main()