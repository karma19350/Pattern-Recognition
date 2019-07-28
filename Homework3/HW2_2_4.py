# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:45:28 2019

@author: Qingyang Zhong
"""

import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(0, 0.05,num=1000)

def parzen(x,h):
  n = len(x)#所有的位置
  y=np.zeros((n,1))
  #print(y)
  for i in np.arange(0,n):
        if x[i] < 0:
            y[i] = 0
        elif x[i] >= 0 and x[i] <= 1:
            y[i] = 1-pow(np.e,-x[i]/h)
        else:
            y[i] = pow(np.e,-x[i]/h)*(pow(np.e,1/h)-1)
  return y

p1=parzen(x,0.002) # h=1
#p2=parzen(x,0.25) # h=1/4
#p3=parzen(x,1/16) # h=1/16

#plt.plot(x1,y1,'y.'); #画图
plt.plot(x,p1,'r',label='h=0.002')
#plt.plot(x,p2,'b',label='h=1/4')
#plt.plot(x,p3,'g',label='h=1/16')
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.title(u"bar P(x)关于x的图像")
plt.legend(loc='lower right')
plt.savefig('./2_2_4.png')