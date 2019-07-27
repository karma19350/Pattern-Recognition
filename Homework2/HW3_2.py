# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 00:33:12 2019

@author: Qingyang Zhong
"""
import os
import random
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score#用pandas加载数据.csv文件，然后用train_test_split分成训练集（75%）和测试集（25%）：
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_curve,auc

iRandom=[]
random_list=list(range(0,10))
random.shuffle(random_list)
iRandom=random_list[0:10]

packet_list=[]
for num in iRandom:
    line='./'+str(num)+'/'
    packet_list.append(line)

data_list=[]

for i in range(len(packet_list)):
    for filename in os.listdir(packet_list[i]):
        img_name=packet_list[i]+filename
        img = io.imread(img_name, as_grey=True)
        arr=img.flatten()
        arr_list=arr.tolist()
        arr_list.append(iRandom[i])
        data_list.append(arr_list)
print(random_list)
print(iRandom)

data=np.array(data_list)
x_train, x_test, y_train, y_test = train_test_split(data[:,0:2304],data[:,-1],test_size=0.25,random_state=0)


#LogisticRegression同样实现了fit()和predict()方法
classifier=LogisticRegression(solver='lbfgs',multi_class ='multinomial')
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)

accuracy=accuracy_score(y_test,y_predict)
print("accuracy：",accuracy)
confusion_matrix=confusion_matrix(y_test,y_predict)
report=classification_report(y_test,y_predict)
print (report)


plt.matshow(confusion_matrix)
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.title(u'混淆矩阵')
plt.colorbar()
plt.ylabel(u'实际类型')
plt.xlabel(u'预测类型')
plt.savefig('./softmax1.png')
plt.show()
print (confusion_matrix)