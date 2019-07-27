# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 00:35:18 2019

@author: Qingyang Zhong
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('breast-cancer-wisconsin.txt',delimiter='\t',header=None)
df.columns=["Sample code number","Clump Thinkness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelisal Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
df.replace(to_replace='?',value=np.nan,inplace=True)
for u in df.columns:
  df[u]=pd.to_numeric(df[u])
#df=df.dropna()#用舍去的方法处理缺省值
new_df=np.array(df)

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score#用pandas加载数据.csv文件，然后用train_test_split分成训练集（75%）和测试集（25%）：
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc

x_train, x_test, y_train, y_test = train_test_split(new_df[:,1:10],new_df[:,10],test_size=0.25)

imp=Imputer(missing_values='NaN',strategy='mean',axis=0)#用均值替代缺省值
#imp=Imputer(missing_values='NaN',strategy='median',axis=0)#用中值替代缺省值
#imp=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)#用众数替代缺省值
x_test=imp.fit_transform(x_test)
x_train=imp.fit_transform(x_train)

#LogisticRegression同样实现了fit()和predict()方法
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
print(y_predict)
scores = cross_val_score(classifier, x_train, y_train, cv=5, scoring='precision')
print("accuracy",np.mean(scores),scores)
confusion_matrix=confusion_matrix(y_test,y_predict)
print (confusion_matrix)


plt.matshow(confusion_matrix)
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.title(u'混淆矩阵')
plt.colorbar()
plt.ylabel(u'实际类型')
plt.xlabel(u'预测类型')
plt.savefig('./average.png')
plt.show()

predictions=classifier.predict_proba(x_test)#每一类的概率
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
, 1])
roc_auc=auc(false_positive_rate,recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig('./roc2.png')
plt.show()