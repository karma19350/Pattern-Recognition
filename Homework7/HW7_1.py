# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:27:28 2019

@author: Qingyang Zhong
"""

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

df = pd.read_excel('题目1.xlsx','Sheet1')
print(df)
data = np.array(df.iloc[:,1:])
print(data)

mds = MDS(dissimilarity='precomputed')
mds.fit(data)
a = mds.embedding_

plt.scatter(a[:,0],a[:,1],color='salmon')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False    
plt.text(a[0,0],a[0,1],u'武汉')
plt.text(a[1,0],a[1,1],u'郑州')
plt.text(a[2,0],a[2,1],u'北京')
plt.text(a[3,0],a[3,1],u'周口')
plt.text(a[4,0],a[4,1],u'运城')
plt.text(a[5,0],a[5,1],u'十堰')
plt.text(a[6,0],a[6,1],u'汉中')
plt.text(a[7,0],a[7,1],u'重庆')
plt.text(a[8,0],a[8,1],u'西安')
plt.text(a[9,0],a[9,1],u'深圳')
plt.savefig('./MDS.png')
#plt.scatter(a[5:10,0],a[5:10,1],color='red')