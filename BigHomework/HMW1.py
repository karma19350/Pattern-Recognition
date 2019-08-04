# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:15:00 2019

@author: Qingyang Zhong
"""
import scipy.io
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#数据预处理
x_data = scipy.io.mmread("train_features.txt").todense()
x_test = scipy.io.mmread("test_features.txt").todense().T
train_labels = pd.read_csv('train_labels.txt',delimiter='\t',header=None)
train_labels=np.array(train_labels)
y_data=[]
for item in train_labels[0]:
    if item=="H1":
        y_data.append(0)
    elif item=="GM":
        y_data.append(1)
    elif item=="K562":
        y_data.append(2)
    elif item=="TF1":
        y_data.append(3)   
    elif item=="HL60":
        y_data.append(4)   
    elif item=="BJ":
        y_data.append(5) 
    elif item=="Leuk":
        y_data.append(6)   
    elif item=="LSC1":
        y_data.append(7)   
    elif item=="Blast":
        y_data.append(8)  
    elif item=="LSC2":
        y_data.append(9)   
    elif item=="LMPP":
        y_data.append(10)   
    elif item=="mono":
        y_data.append(11)   
y_data=np.array(y_data)
y_data=y_data.reshape(1,1000)
y_df=pd.DataFrame(y_data)
x_df=pd.DataFrame(x_data)
#print(x_test.shape)
raw_data=np.array(pd.concat([x_df,y_df],ignore_index=False))
data=raw_data.T


  
'''****************************************************************************'''
#相关系数法
def takeOrder(elem):
    return elem[1]
def CorrSelection(x_data,y_data,x_val):
    Corr=[]
    length=x_data.shape[1]
    for i in range(length):
        if all(x_data[:,i]==0):
            continue
        cor=np.corrcoef(x_data[:,i].T,y_data)
        Corr.append([i,abs(cor[0][1])])
    Corr.sort(key=takeOrder,reverse = True)
    Corr=np.array(Corr)
    featureList=Corr[0:1000,0].astype(int)
    #print("correlation coefficient feature:",featureList)
    x_data_new=x_data[:,featureList[0]].reshape(800,1)
    x_val_new = x_val[:,featureList[0]].reshape(200,1)
    for i in range(len(featureList)):
        if i==0:
            continue
        x_data_new=np.concatenate((x_data_new,x_data[:,featureList[i]].reshape(800,1)),axis=1)
        x_val_new =np.concatenate((x_val_new,x_val[:,featureList[i]].reshape(200,1)),axis=1)
    print(x_data_new.shape)
    return x_data_new,x_val_new

#阈值选择法 
from sklearn.feature_selection import VarianceThreshold
def VarianceThre(x_train,x_test):
    sel = VarianceThreshold(threshold=0.98)
    x_train_new=sel.fit_transform(x_train)
    x_test_new=sel.transform(x_test)
    print(x_test_new.shape[1])
    return x_train_new,x_test_new

#PCA降维处理
from sklearn.decomposition import PCA
def PCA_Reduction(x_train,x_val):
    #pca1 = PCA(n_components=0.98)
    pca1 = PCA(n_components=100)
    pca1.fit(x_train)
    x_val_new = pca1.transform(x_val)
    x_train_new = pca1.transform(x_train)
    return x_train_new,x_val_new

# filtering特征选择
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
def filtering(x_train,y_train,x_val):
    model=SelectKBest(f_classif,k=2000)
    label=model.fit(x_train,y_train).get_support(indices=True)
    x_train_new=model.transform(x_train)
    x_val_new=x_val[:,np.transpose(label)]
    return x_train_new,x_val_new
'''****************************************************************************'''
#SVM法
from sklearn.svm import SVC  
from sklearn.linear_model.logistic import LogisticRegression
def SVM_Classify(x_train,y_train,x_test,y_test):
    #clf = SVC(kernel='poly', degree=3,gamma=10,verbose=1,decision_function_shape='ovo')
    clf = SVC(kernel='linear',verbose=1)
    clf.fit(x_train, y_train) 
    y_predition_test=clf.predict(x_test)
    total=0
    right=0
    for i in range(len(y_predition_test)):
        if y_predition_test [i]==y_test[i]:
            right+=1
        total+=1
    acc=float(right/total)   
    print('SVM val accuarcy: ' + str(acc))
    return acc

#Logistic Regression方法    
def Logistic_Regression(x_train,y_train,x_test,y_test):
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
    print('Logistic Regression val accuarcy: ' + str(acc))
    return acc

#神经网络方法   
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
def NeuralNetwork(x_train,y_train,x_val,y_val):
    y_train_nn=to_categorical(y_train)
    y_val_nn=to_categorical(y_val)
    model = Sequential()
    model.add(Dense(100, input_shape=(x_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dense(12))
    model.add(Activation('softmax'))
    fBestModel = 'best_model.h5' 
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1) 
    best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) 
    model.compile(optimizer= sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    history=model.fit(x_train, y_train_nn, validation_data = (x_val, y_val_nn), epochs=20, 
          batch_size=128, callbacks=[best_model, early_stop]) 
    model.summary()
    Y_pre = np.argmax(model.predict(x_val),1)
    Y_test = np.argmax(y_val_nn,1)
    total=0
    right=0
    for i in range(len(Y_pre)):
        if Y_pre[i]==Y_test[i]:
            right+=1
        total+=1
    acc=float(right/total)   
    print('nn test accuarcy: ' + str(acc))
    confusion_mat=confusion_matrix(Y_test,Y_pre)
    plt.figure(1)
    plt.matshow(confusion_mat,cmap=plt.cm.Blues)
    plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    indices = range(len(confusion_mat))
    plt.xticks(indices)
    plt.yticks(indices)
    plt.colorbar()
    plt.ylabel(u'实际类型')
    plt.xlabel(u'预测类型')
    for first_index in range(12):
        for second_index in range(12):
            plt.text(first_index, second_index, confusion_mat[second_index][first_index], va='center', ha='center')
    plt.savefig('./matrix_nn.png')
    plt.show()
    return acc

#RandomForest方法
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import classification_report
def RandomForest(x_train_new,y_train,x_val_new):
    '''parameters={
            'criterion':['gini','entropy'],
            'max_depth':[50,80,100,120],
            'min_impurity_decrease':[0,0.005,0.001]
            }
    dtree=RandomForestClassifier()
    grid_search=GridSearchCV(dtree,parameters,scoring='accuracy',cv=4)
    grid_search.fit(x_train_new,y_train)
    grid_search.best_estimator_   #查看grid_search方法 
    grid_search.best_score_       #正确率 
    grid_search.best_params_      #最佳 参数组合
    print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))'''
    dtree=RandomForestClassifier(criterion='gini',max_depth=120,min_impurity_decrease=0) 
    dtree.fit(x_train_new,y_train) 
    pred=dtree.predict(x_val_new) 
    print(classification_report(y_val,pred))
'''****************************************************************************'''
# 使用TSNE进行降维处理
from sklearn.manifold import TSNE
def TSNE_Show(x_val_new,y_val_new):
    tsne = TSNE(n_components=2, learning_rate=50,verbose=1)
    tsne_model = tsne.fit_transform(x_val_new)
    plt.figure(2)
    plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=y_val_new)
    plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    plt.rcParams['axes.unicode_minus']=False   
    plt.title('t-SNE Visualization')
    plt.savefig('./t-SNE_show.png')
    return tsne_model

# 使用lle进行jianw降维处理
from sklearn.manifold import LocallyLinearEmbedding
def LLE_Show(x_data_new,y_data_new):
    lle = LocallyLinearEmbedding(n_components=2,n_neighbors=10)
    lle_model = lle.fit_transform(x_data_new)
    plt.figure(6)
    plt.scatter(lle_model[:, 0], lle_model[:, 1], c=y_data_new)
    plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    plt.rcParams['axes.unicode_minus']=False   
    plt.title('LLE Visualization')
    plt.savefig('./lle_show.png')
    return lle_model

#使用PCA进行可视化
def PCA_Show(x_train,y_train):
    pca = PCA(n_components=2)
    pca_model=pca.fit_transform(x_train)
    plt.figure(5)
    plt.scatter(pca_model[:, 0], pca_model[:, 1], c=y_train)
    plt.title('PCA Visualization')
    plt.savefig('./pca_show.png')
    plt.show()
    return pca_model
'''****************************************************************************'''
#KMeans聚类   
from sklearn.cluster import KMeans
def KMeans_Cluster(model):
    y_pred = KMeans(n_clusters=12).fit_predict(model)
    plt.figure(7)
    plt.scatter(model[:, 0], model[:, 1], c=y_pred)
    plt.title('KMeans Visualization')
    plt.legend(loc='lower right') 
    plt.savefig('./KMeans.png')
    plt.show()
    
#5折交叉验证   
'''from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)

VarianceThre_SVM_list=[]
VarianceThre_Logistic_Regression_list=[]
VarianceThre_NeuralNetwork_list=[]
VarianceThre_RandomForest_list=[]

CorrSelection_SVM_list=[]
CorrSelection_Logistic_Regression_list=[]
CorrSelection_NeuralNetwork_list=[]
CorrSelection_RandomForest_list=[]

PCA_Reduction_SVM_list=[]
PCA_Reduction_Logistic_Regression_list=[]
PCA_Reduction_NeuralNetwork_list=[]
PCA_Reduction_RandomForest_list=[]

for train_index, test_index in kf.split(data):
    x_train=data[train_index,:-1]
    y_train=data[train_index,-1]
    x_val=data[test_index,:-1]
    y_val=data[test_index,-1]
    print("********************************************************")
    x_train_new,x_val_new=VarianceThre(x_train,x_val)
    VarianceThre_SVM_list.append(SVM_Classify(x_train_new,y_train,x_val_new,y_val))
    VarianceThre_Logistic_Regression_list.append(Logistic_Regression(x_train_new,y_train,x_val_new,y_val))
    VarianceThre_NeuralNetwork_list.append(NeuralNetwork(x_train_new,y_train,x_val_new,y_val))
    VarianceThre_RandomForest_list.append(RandomForest(x_train_new,y_train,x_val_new))
    
    x_train_new,x_val_new =CorrSelection(x_train,y_train,x_val)
    CorrSelection_SVM_list.append(SVM_Classify(x_train_new,y_train,x_val_new,y_val))
    CorrSelection_Logistic_Regression_list.append(Logistic_Regression(x_train_new,y_train,x_val_new,y_val))
    CorrSelection_NeuralNetwork_list.append(NeuralNetwork(x_train_new,y_train,x_val_new,y_val))
    CorrSelection_RandomForest_list.append(RandomForest(x_train_new,y_train,x_val_new))
    
    x_train_new,x_val_new =PCA_Reduction(x_train,x_val)
    PCA_Reduction_SVM_list.append(SVM_Classify(x_train_new,y_train,x_val_new,y_val))
    PCA_Reduction_Logistic_Regression_list.append(Logistic_Regression(x_train_new,y_train,x_val_new,y_val))
    PCA_Reduction_NeuralNetwork_list.append(NeuralNetwork(x_train_new,y_train,x_val_new,y_val))
    PCA_Reduction_RandomForest_list.append(RandomForest(x_train_new,y_train,x_val_new))

print("*************************************************************")
print("VarianceThre_SVM_list:",np.mean(VarianceThre_SVM_list),VarianceThre_SVM_list)
print("VarianceThre_Logistic_Regression_list:",np.mean(VarianceThre_Logistic_Regression_list),VarianceThre_Logistic_Regression_list)
print("VarianceThre_NeuralNetwork_list:",np.mean(VarianceThre_NeuralNetwork_list),VarianceThre_NeuralNetwork_list)
#print("VarianceThre_RandomForest_list:",np.mean(VarianceThre_RandomForest_list),VarianceThre_RandomForest_list)
         
print("CorrSelection_SVM_list:",np.mean(CorrSelection_SVM_list),CorrSelection_SVM_list)
print("CorrSelection_Logistic_Regression_list:",np.mean(CorrSelection_Logistic_Regression_list),CorrSelection_Logistic_Regression_list)
print("CorrSelection_NeuralNetwork_list:",np.mean(CorrSelection_NeuralNetwork_list),CorrSelection_NeuralNetwork_list)
#print("CorrSelection_RandomForest_list:",np.mean(CorrSelection_RandomForest_list),CorrSelection_RandomForest_list)

print("PCA_Reduction_SVM_list:",np.mean(PCA_Reduction_SVM_list),PCA_Reduction_SVM_list)
print("PCA_Reduction_Logistic_Regression_list:",np.mean(PCA_Reduction_Logistic_Regression_list),PCA_Reduction_Logistic_Regression_list)
print("PCA_Reduction_NeuralNetwork_list:",np.mean(PCA_Reduction_NeuralNetwork_list),PCA_Reduction_NeuralNetwork_list)
#print("PCA_Reduction_RandomForest_list:",np.mean(PCA_Reduction_RandomForest_list),PCA_Reduction_RandomForest_list) 
'''    
#x_train_new,x_val_new=VarianceThre(x_train,x_val)
#x_train_new,x_val_new =CorrSelection(x_train,y_train,x_val)

#对测试集进行预测
from sklearn.model_selection import train_test_split
x_data=data[:,:-1]
y_data=data[:,-1]

prediction_list=[]
for i in range(5):
   x_train, x_val, y_train, y_val = train_test_split(data[:,:-1], data[:,-1], test_size=0.2)
   x_train_new,x_test_new =PCA_Reduction(x_train,x_test)
   print(x_test_new.shape)
   clf = SVC(kernel='linear',verbose=1)
   clf.fit(x_train_new, y_train) 
   y_predition_test=clf.predict(x_test_new)
   prediction_list.append(y_predition_test)
   
   classifier=LogisticRegression()
   classifier.fit(x_train_new,y_train)
   y_predict=classifier.predict(x_test_new)
   prediction_list.append(y_predict)
   
   dtree=RandomForestClassifier(criterion='gini',max_depth=120,min_impurity_decrease=0) 
   dtree.fit(x_train_new,y_train) 
   pred=dtree.predict(x_test_new) 
   prediction_list.append(pred)
   
   
print(prediction_list)
prediction_result=np.array(prediction_list).T 
print(prediction_result.shape)
test_result=[]
for x in prediction_result:
    test_result.append(np.argmax(np.bincount(x)))
print(test_result)
y_test=[]
for item in test_result:
    if item==0:
        y_test.append("H1")
    elif item==1:
        y_test.append("GM")
    elif item==2:
        y_test.append("K562")
    elif item==3:
        y_test.append("TF1")   
    elif item==4:
        y_test.append("HL60")   
    elif item==5:
        y_test.append("BJ") 
    elif item==6:
        y_test.append("Leuk")   
    elif item==7:
        y_test.append("LSC1")   
    elif item==8:
        y_test.append("Blast")  
    elif item==9:
        y_test.append("LSC2")   
    elif item==10:
        y_test.append("LMPP")   
    elif item==11:
        y_test.append("mono")   
with codecs.open('problem1.txt','a')as f1:
    for item in y_test:
        f1.write(item+'\t')
   
'''
#可视化与聚类
x_train, x_val, y_train, y_val = train_test_split(data[:,:-1], data[:,-1], test_size=0.2)
x_data_new,x_val_new =PCA_Reduction(x_data,x_val)
PCA_model=PCA_Show(x_data_new,y_data)
KMeans_Cluster(PCA_model)
#TSNE_model=TSNE_Show(x_data_new,y_data)
#KMeans_Cluster(TSNE_model)
#lle_model=LLE_Show(x_data_new,y_data)
#KMeans_Cluster(lle_model)'''


