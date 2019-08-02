# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:06:33 2019

@author: Qingyang Zhong
"""
import scipy.io as scio 
dataFile = 'Sogou_webpage.mat'  
data = scio.loadmat(dataFile)  
feature = data['wordMat'] 
label = data['doclabel']
label = label[:,0]


import numpy as np
from sklearn import tree
from sklearn import model_selection 
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=3)

parameters={
            'criterion':['gini','entropy'],
            'max_depth':[50,80,100,120],
            'min_impurity_decrease':[0,0.05,0.1]
            }
dtree=tree.DecisionTreeClassifier()
grid_search=GridSearchCV(dtree,parameters,scoring='accuracy',cv=4)
grid_search.fit(x_train,y_train)
grid_search.best_estimator_   #查看grid_search方法 
grid_search.best_score_       #正确率 
grid_search.best_params_      #最佳 参数组合

print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))


dtree=tree.DecisionTreeClassifier(criterion='gini',max_depth=80,min_impurity_decrease=0) 
dtree.fit(x_train,y_train) 
pred=dtree.predict(x_test) 
print(classification_report(y_test,pred))
