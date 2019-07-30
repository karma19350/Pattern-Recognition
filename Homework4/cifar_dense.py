# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:06:53 2019

@author: Qingyang Zhong
"""

import numpy as np
import matplotlib.pyplot as plt 
npzfile = np.load('cifar10.npz') 
x_train = npzfile['x_train']
x_test = npzfile['x_test']
y_train = npzfile['y_train']
y_test = npzfile['y_test']
y_train = y_train[:,0:3]
y_test = y_test[:,0:3]
#print(x_test)

#数据预处理
dims = x_train.shape[0]
x_train=x_train.reshape(-1,3072)

dims_test = x_test.shape[0]
x_test=x_test.reshape(-1,3072)

from sklearn.model_selection import train_test_split
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
x_test_new, x_val, y_test_new, y_val = train_test_split(x_test_new, y_test_new, test_size=0.5, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import metrics

'''0层隐层'''
model = Sequential()
model.add(Dense(3, input_shape=(3072,)))
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.1, decay=1e-6) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x_train_new, y_train_new,validation_data = (x_val, y_val),epochs = 20)
model.summary()
# calculate accuracy
Y_pre = np.argmax(model.predict(x_test),1)
Y_test = np.argmax(y_test,1)
acc = sum(Y_pre == Y_test)/len(Y_pre)
print('test accuarcy: ' + str(acc))

#plt.subplot(2,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.legend(['acc','val_acc','loss','val_loss'], loc='lower right')
plt.savefig('./curve0.png')
plt.show()

'''1层隐层'''
model = Sequential()
model.add(Dense(512, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.1, decay=1e-6) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x_train_new, y_train_new,validation_data = (x_val, y_val),epochs = 20)
model.summary()
# calculate accuracy
Y_pre = np.argmax(model.predict(x_test),1)
Y_test = np.argmax(y_test,1)
acc = sum(Y_pre == Y_test)/len(Y_pre)
print('test accuarcy: ' + str(acc))

#plt.subplot(2,2,2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.legend(['acc','val_acc','loss','val_loss'], loc='lower right')
plt.savefig('./curve1.png')
plt.show()

'''2层隐层'''
model = Sequential()
model.add(Dense(512, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.1, decay=1e-6) 
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train_new, y_train_new,validation_data = (x_val, y_val),epochs = 20)
model.summary()
# calculate accuracy
Y_pre = np.argmax(model.predict(x_test),1)
Y_test = np.argmax(y_test,1)
acc = sum(Y_pre == Y_test)/len(Y_pre)
print('test accuarcy: ' + str(acc))

#plt.subplot(2,2,2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.legend(['acc','val_acc','loss','val_loss'], loc='lower right')
plt.savefig('./curve2.png')
plt.show()

'''3层隐层'''
model = Sequential()
model.add(Dense(512, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.1, decay=1e-6) 
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train_new, y_train_new,validation_data = (x_val, y_val),epochs = 20)
model.summary()
# calculate accuracy
Y_pre = np.argmax(model.predict(x_test),1)
Y_test = np.argmax(y_test,1)
acc = sum(Y_pre == Y_test)/len(Y_pre)
print('test accuarcy: ' + str(acc))
#plt.subplot(2,2,2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.legend(['acc','val_acc','loss','val_loss'], loc='lower right')
plt.savefig('./curve3.png')
plt.show()
