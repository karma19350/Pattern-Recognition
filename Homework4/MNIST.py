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

# 方法2：直接从文件夹中载入
'''import numpy as np
npzfile = np.load('mnist.npz') 
X_train = npzfile['X_train']
X_test = npzfile['X_test']
y_train = npzfile['y_train']
y_test = npzfile['y_test']'''
'''plt.subplot(1,10,2)
plt.imshow(X_test[1].squeeze(),cmap='gray')
plt.title('%i'% y_test[1])
plt.savefig('./softmax1.png')
plt.show()'''
#画图
for i in range(10):
    pos=0
    for j in y_test:
        if j==i:
            plt.subplot(1,10,i+1)
            plt.imshow(X_test[pos].squeeze(),cmap='gray')
            plt.title('%i'% y_test[pos])
            plt.axis('off')
            break
        else:
            pos+=1
plt.savefig('./show_all.png')
plt.show()         
#数据预处理
dims = X_train.shape[0]
x_train=X_train.reshape(-1,784)
x_train=x_train.astype(np.float32)
x_train/=255.0

dims_test = X_test.shape[0]
x_test=X_test.reshape(-1,784)
x_test=x_test.astype(np.float32)
x_test/=255.0

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras import optimizers

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model = Sequential()
model.add(Dense(100, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
'''sgd = optimizers.SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(x_train, y_train,epochs = 20)
model.summary()'''

from sklearn.model_selection import train_test_split
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
x_test_new, x_val, y_test_new, y_val = train_test_split(x_test_new, y_test_new, test_size=0.5, random_state=42)

from keras.callbacks import EarlyStopping, ModelCheckpoint
# 加入early_stop与best_model选项
fBestModel = 'best_model.h5' 
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1) 
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(optimizer= sgd, loss='categorical_crossentropy')
history=model.fit(x_train_new, y_train_new, validation_data = (x_val, y_val), epochs=20, 
          batch_size=128, callbacks=[best_model, early_stop]) 
model.summary()

# calculate accuracy
Y_pre = np.argmax(model.predict(x_test),1)
Y_test = np.argmax(y_test,1)
acc = sum(Y_pre == Y_test)/len(Y_pre)
print('test accuarcy: ' + str(acc))

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./curve100.png')
plt.show()

#calculate confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(Y_test,Y_pre)
print (confusion_matrix)
plt.matshow(confusion_matrix,cmap=plt.cm.Blues)
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
indices = range(len(confusion_matrix))
plt.xticks(indices)
plt.yticks(indices)
plt.colorbar()
plt.ylabel(u'实际类型')
plt.xlabel(u'预测类型')
for first_index in range(10):
    for second_index in range(10):
        plt.text(first_index, second_index, confusion_matrix[second_index][first_index], va='center', ha='center')
plt.savefig('./matrix100.png')
plt.show()

pos=0
i=0
for j in range(len(Y_pre)):
    if Y_pre[j]==4 and Y_test[j]==9:
        plt.subplot(2,3,i+1)
        plt.imshow(X_test[pos].squeeze(),cmap='gray')
        plt.title('label:%i,pred:%i'% (Y_test[pos],Y_pre[pos]))
        plt.axis('off')
        i+=1
    if i>=3:
        break
    pos+=1
pos=0
i=3
for j in range(len(Y_pre)):
    if Y_pre[j]==9 and Y_test[j]==4:
        plt.subplot(2,3,i+1)
        plt.imshow(X_test[pos].squeeze(),cmap='gray')
        plt.title('label:%i,pred:%i'% (Y_test[pos],Y_pre[pos]))
        plt.axis('off')
        i+=1
    if i>=6:
        break
    pos+=1
plt.savefig('./被认错的9与4.png')
plt.show()   