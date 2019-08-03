# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:11:21 2019

@author: Qingyang Zhong
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
import numpy as np

#数据预处理
dims = X_train.shape[0]
x_train=X_train.reshape(-1,784)
x_train=x_train.astype(np.float32)
x_train/=255.0

dims_test = X_test.shape[0]
x_test=X_test.reshape(-1,784)
x_test=x_test.astype(np.float32)
x_test/=255.0


x_data=[]
y_data=[]
for i in range(len(x_train)):
    if y_train[i]==0:
        x_data.append(x_train[i])
        y_data.append(0)
    if y_train[i]==8:
        x_data.append(x_train[i])
        y_data.append(8)
for i in range(len(x_test)):
    if y_test[i]==0:
        x_data.append(x_test[i])
        y_data.append(0)
    if y_test[i]==8:
        x_data.append(x_test[i])
        y_data.append(8)
        
x_data_new=np.array(x_data)
y_data_new=np.array(y_data)

from sklearn.model_selection import train_test_split
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_data_new, y_data_new, test_size=0.3, random_state=3)

from sklearn.decomposition import PCA

# 使用PCA 进行降维处理
pca = PCA(n_components=10)
pca_model=pca.fit_transform(x_train_new)
plt.figure()
plt.scatter(pca_model[:, 0], pca_model[:, 1], c=y_train_new)
plt.title('PCA Visualization')
plt.savefig('./pca.png')
a= pca.explained_variance_ratio_
print(a)  
plt.figure(num=2, figsize=(8, 5)) 
x=[1,2,3,4,5,6,7,8,9,10]
plt.plot(x,a,'*-')
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.xlabel('特征数')
plt.ylabel('解释方差比例')
plt.xticks(x)
plt.savefig('./ratio.png')
plt.show()

pca1 = PCA(n_components=0.98)
pca_model2=pca1.fit(x_train_new)

a1= pca1.explained_variance_ratio_
print(a1.shape)  
plt.figure(num=4, figsize=(8, 5)) 
x1=range(len(a1))
plt.plot(x1,a1,'*-')
plt.xlabel('特征数')
plt.ylabel('解释方差比例')
plt.savefig('./ratio1.png')

X_test_new = pca1.transform(x_test_new)
X_train_new = pca1.transform(x_train_new)
plt.figure(num=5, figsize=(8, 5)) 
plt.scatter(X_test_new[:, 0], X_test_new[:, 1],c=y_test_new)
plt.title('test set PCA visualization')
plt.savefig('./pca2.png')
plt.show()

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras import optimizers

y_train_nn=to_categorical(y_train_new)
y_test_nn=to_categorical(y_test_new)
x_train_nn, x_val, y_train_nn, y_val = train_test_split(X_train_new, y_train_nn, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(100, input_shape=(len(a1),)))
model.add(Activation('relu'))
model.add(Dense(9))
model.add(Activation('softmax'))

from keras.callbacks import EarlyStopping, ModelCheckpoint
fBestModel = 'best_model.h5' 
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1) 
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(optimizer= sgd, loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train_nn, y_train_nn, validation_data = (x_val, y_val), epochs=20, 
          batch_size=128, callbacks=[best_model, early_stop]) 
model.summary()

# calculate accuracy
Y_pre = np.argmax(model.predict(X_test_new),1)
Y_test = np.argmax(y_test_nn,1)
total=0
right=0
for i in range(len(Y_pre)):
    if Y_pre[i]==Y_test[i]:
        right+=1
    total+=1
acc=float(right/total)   
print('nn test accuarcy: ' + str(acc))

#calculate confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(Y_test,Y_pre)
plt.matshow(confusion_matrix,cmap=plt.cm.Blues)
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
indices = range(len(confusion_matrix))
plt.xticks(indices)
plt.yticks(indices)
plt.colorbar()
plt.ylabel(u'实际类型')
plt.xlabel(u'预测类型')
for first_index in range(2):
    for second_index in range(2):
        plt.text(first_index, second_index, confusion_matrix[second_index][first_index], va='center', ha='center')
plt.savefig('./matrixPCA.png')
plt.show()