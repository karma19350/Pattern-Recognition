# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:30:52 2019

@author: wangpeng884112
"""

# 这里提供了载入MNIST数据集的两种方式
# 方法1:调用keras函数载入
from keras.datasets import mnist
import matplotlib.pyplot as plt 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
import numpy as np
#npzfile = np.savez('mnist.npz',X_train = X_train,  
#                   y_train =  y_train, X_test = X_test, y_test = y_test) 

#数据预处理
dims = X_train.shape[0]
x_train=X_train.reshape(-1,784)
x_train=x_train.astype(np.float32)
x_train/=255.0

dims_test = X_test.shape[0]
x_test=X_test.reshape(-1,784)
x_test=x_test.astype(np.float32)
x_test/=255.0

#将标签处理为-1,1
x_data=[]
y_data=[]

for i in range(len(x_train)):
    if y_train[i]==4:
        x_data.append(x_train[i])
        y_data.append(1)
    if y_train[i]==9:
        x_data.append(x_train[i])
        y_data.append(-1)
for i in range(len(x_test)):
    if y_test[i]==4:
        x_data.append(x_test[i])
        y_data.append(1)
    if y_test[i]==9:
        x_data.append(x_test[i])
        y_data.append(-1)
        
x_data_new=np.array(x_data)
y_data_new=np.array(y_data)

from sklearn.model_selection import train_test_split
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_data_new, y_data_new, test_size=0.3, random_state=3)

#支持向量机方法
from sklearn.svm import SVC   
clf = SVC(kernel='linear',verbose=1)
#clf = SVC(kernel='poly', degree=2,gamma=10,verbose=1)
#clf = SVC(kernel='poly', degree=3,gamma=10,verbose=1)
#clf = SVC(kernel='sigmoid', gamma=0.005,verbose=1)
#clf = SVC(kernel='rbf', gamma=1,verbose=1)         
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
print('linear train accuarcy: ' + str(acc))
total=0
right=0
for i in range(len(y_predition_test)):
    if y_predition_test [i]==y_test_new[i]:
        right+=1
    total+=1
acc=float(right/total)   
print('linear test accuarcy: ' + str(acc))
print(clf.n_support_)

#将标签处理为-1,1
y_data_nn=[]

for i in range(len(x_train)):
    if y_train[i]==4:
        y_data_nn.append(1)
    if y_train[i]==9:
        y_data_nn.append(0)
for i in range(len(x_test)):
    if y_test[i]==4:
        y_data_nn.append(1)
    if y_test[i]==9:
        y_data_nn.append(0)
        
y_data_nn_new=np.array(y_data_nn)
x_train_new, x_test_new, y_train_new1, y_test_new1 = train_test_split(x_data_new, y_data_nn_new, test_size=0.3, random_state=3)

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix

classifier=LogisticRegression()
classifier.fit(x_train_new,y_train_new1)
y_predict=classifier.predict(x_test_new)
precisions = cross_val_score(classifier, x_train_new, y_train_new1, cv=5, scoring='precision')
print(precisions)
print(np.mean(precisions))

confusion_matrix=confusion_matrix(y_test_new1,y_predict)

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

#神经网络方法
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras import optimizers

y_train_nn=to_categorical(y_train_new1)
y_test_nn=to_categorical(y_test_new1)
x_train_nn, x_val, y_train_nn, y_val = train_test_split(x_train_new, y_train_nn, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(100, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

from keras.callbacks import EarlyStopping, ModelCheckpoint
fBestModel = 'best_model.h5' 
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1) 
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(optimizer= sgd, loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(x_train_nn, y_train_nn, validation_data = (x_val, y_val), epochs=20, 
          batch_size=128, callbacks=[best_model, early_stop]) 
model.summary()

# calculate accuracy
Y_pre = np.argmax(model.predict(x_test_new),1)
Y_test = np.argmax(y_test_nn,1)
total=0
right=0
for i in range(len(Y_pre)):
    if Y_pre[i]==Y_test[i]:
        right+=1
    total+=1
acc=float(right/total)   
print('nn test accuarcy: ' + str(acc))

# summarize history for loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy and loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.legend(['acc','val_acc','loss','val_loss'], loc='lower right')
plt.savefig('./curve100.png')
plt.show()

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
plt.savefig('./matrix100.png')
plt.show()

