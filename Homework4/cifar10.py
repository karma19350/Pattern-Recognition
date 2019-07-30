
import numpy as np
import matplotlib.pyplot as plt 

npzfile = np.load('cifar10.npz') 
x_train = npzfile['x_train']
x_test = npzfile['x_test']
y_train = npzfile['y_train']
y_test = npzfile['y_test']
y_train = y_train[:,0:3]
y_test = y_test[:,0:3]

from sklearn.model_selection import train_test_split
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
x_test_new, x_val, y_test_new, y_val = train_test_split(x_test_new, y_test_new, test_size=0.5, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
nb_classes=3

shape_ord = x_train_new.shape[1:]

model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding = 'valid',input_shape = (32,32,3)))
model.add(Activation('relu'))
#model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
history=model.fit(x_train_new, y_train_new,validation_data = (x_val, y_val),epochs = 20)
model.summary()

# calculate accuracy
Y_pre = np.argmax(model.predict(x_test),1)
Y_test = np.argmax(y_test,1)
acc = sum(Y_pre == Y_test)/len(Y_pre)
print('test accuarcy: ' + str(acc))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.legend(['acc','val_acc','loss','val_loss'], loc='lower right')
#plt.savefig('./curve_nn.png')
#plt.savefig('./curve_sigmoid.png')
plt.savefig('./curve_no_dropout.png')
plt.show()