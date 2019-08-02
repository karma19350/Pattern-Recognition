# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:43:14 2019

@author: Qingyang Zhong
"""

import scipy.io as scio 
dataFile = 'Sogou_webpage.mat'  
data = scio.loadmat(dataFile)  
feature = data['wordMat'] 
label = data['doclabel']
#label = label[:,0]

import math
import numpy as np
import time
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=3)
#x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=3)
data_train=np.concatenate((x_train,y_train),axis=1)
data_test=np.concatenate((x_test,y_test),axis=1)
#data_val=np.concatenate((x_val,y_val),axis=1)
y_test = y_test[:,0]
#print(data_test.shape)

# 计算数据的熵(entropy) 信息不纯度
def Impurtity(samples): 
    numEntries=len(samples) # 数据条数 
    labelCounts={} 
    for featVec in samples: 
        if len(featVec)==0:
            return 0
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别） 
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel]=0 
        labelCounts[currentLabel]+=1 # 统计有多少个类以及每个类的数量 
    shannonEnt=0 
    for key in labelCounts: 
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值 
        shannonEnt-=prob*math.log(prob,2) # 累加每个类的熵值 
    return shannonEnt

def SplitNode(SamplesUnderThisNode, axis, value):#划分节点的阈值设定挪入GenerateTree中
     retDataSet = []
     reducedFeatVec=[]
     for featVec in SamplesUnderThisNode:
         if type(featVec)!=type([]):
             featVec=featVec.tolist()
         if featVec[axis] == value:
             reducedFeatVec = featVec[:axis]
             reducedFeatVec.extend(featVec[axis+1:])#把当前选中的特征从其中删掉
             retDataSet.append(reducedFeatVec)
     return retDataSet
     
def SelectFeature(SamplesUnderThisNode):
     numFeatures = len(SamplesUnderThisNode[0]) - 1#因为数据集的最后一项是标签,该节点下的特征数
     baseEntropy = Impurtity(SamplesUnderThisNode)#未扩展前的信息熵
     bestInfoGain = 0.0
     bestFeature = -1
     for i in range(numFeatures):
         featList = [example[i] for example in SamplesUnderThisNode]
         uniqueVals = set(featList)#统计该特征有多少种取值
         newEntropy = 0.0
         for value in uniqueVals:
             subSamplesUnderThisNode = SplitNode(SamplesUnderThisNode, i, value)#特征所在维度与特征值0/1
             prob = len(subSamplesUnderThisNode) / float(len(SamplesUnderThisNode))
             newEntropy += prob * Impurtity(subSamplesUnderThisNode)
         infoGain = baseEntropy -newEntropy
         if infoGain > bestInfoGain:
             bestInfoGain = infoGain
             bestFeature = i
     return bestFeature


#多数表决的方式计算节点分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)        
 
def GenerateTree(dataSet, labels,thre,max_depth):#递归方法构建树
     classList = [example[-1] for example in dataSet]
     if classList.count(classList[0]) ==len(classList):#类别相同则停止划分
         return classList[0]
     if Impurtity(dataSet)<thre:#不纯度已小于阈值
         return majorityCnt(classList)
     if len(dataSet[0]) == 1:#所有特征已经用完
         return majorityCnt(classList)
     if max_depth <= 0:#以达到树的深度
         return majorityCnt(classList)
     bestFeat = SelectFeature(dataSet)
     bestFeatLabel = labels[bestFeat]
     #print(bestFeatLabel)
     myTree = {bestFeatLabel:{}}
     del(labels[bestFeat])
     featValues = [example[bestFeat] for example in dataSet]
     uniqueVals = set(featValues)
     for value in uniqueVals:
         subLabels = labels[:]#为了不改变原始列表的内容复制了一下
         myTree[bestFeatLabel][value] = GenerateTree(SplitNode(dataSet, 
                                         bestFeat, value),subLabels,thre,max_depth-1)
     return myTree

def Decision(GeneratedTree,SampleToBePredicted,featLabels):
    currentFeat = list(GeneratedTree.keys())[0]
    #print(currentFeat)
    secondTree = GeneratedTree[currentFeat]
    featureIndex = featLabels.index(currentFeat)
    classLabel=1
    for value in secondTree.keys():
        if value == SampleToBePredicted[featureIndex]:
            if type(secondTree[value]).__name__ == 'dict':
                classLabel = Decision(secondTree[value],SampleToBePredicted,featLabels)
            else:
                classLabel = secondTree[value]
    return classLabel
    #except AttributeError:
        #print(secondTree)


label = [x for x in range(1201) if x > 0]
labels = [x for x in range(1201) if x > 0]
threshold=[0,0.5,1]
max_depth=[30,80,50]

from sklearn.model_selection import KFold

def main():
  kf = KFold(n_splits=4)
  for thre in threshold:
    for depth in max_depth:
        accuracy_list=[]
        for train_index, test_index in kf.split(data_train):
            train_cv=data_train[train_index]
            test_cv=data_train[test_index]
            x_test_cv=test_cv[:,:-1]
            y_test_cv=test_cv[:,-1]
            
            t1 = time.clock()
            myTree = GenerateTree(train_cv,label,thre,depth)
            t2 = time.clock()
            print ('Complete establishing! Execute for ',t2-t1)
            
            y_pred=[]
            for test in x_test_cv:
                temp=Decision(myTree,test,labels)
                y_pred.append(temp)
                #print(temp,y_test[j])
            right=0
            for i in range(len(y_test_cv)):
                 if y_pred[i]==y_test_cv[i]:
                     right = right+1
            accuracy=right/ float(len(y_test_cv))
            print("accuracy:",accuracy)
            accuracy_list.append(accuracy)
        accuracy_list=np.array(accuracy_list)
        print("Threshold:",thre,"  Max_deoth:",depth)
        print("Average accuracy:",np.mean(accuracy_list))
        
  t3 = time.clock()
  myTree = GenerateTree(data_train,label,0,80)
  t4 = time.clock()
  print ('Complete establishing! Execute for ',t4-t3)
  y_pred=[]
  for test in x_test:
      temp=Decision(myTree,test,labels)
      y_pred.append(temp)
  right=0
  for i in range(len(y_test)):
      if y_pred[i]==y_test[i]:
           right=right+1
  accuracy=right/ float(len(y_test))
  print("accuracy:",accuracy)
  
  
if __name__=='__main__':
    main()