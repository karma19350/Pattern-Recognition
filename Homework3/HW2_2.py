# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:57:53 2019

@author: Qingyang Zhong
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as st

xi = np.random.rand(256)#样本点
x = np.linspace(0, 1,num=1000)

def phi(x, xi, h):
# 判断x, xi之间的距离
    if xi < x:
        phi = pow(np.e,-(x-xi)/h);
    else:
        phi = 0;
    return phi

def parzen(xi,x,h):
  n = len(x)#所有的位置
  y=np.zeros((n,1))
  #print(y)
  for i in np.arange(0,n):
    m = 0
    for j in np.arange(0,len(xi)):
        m = m + phi(x[i], xi[j], h)
    y[i] = m / (len(xi) * h)
  return y

p1=parzen(xi,x,1) # h=1
p2=parzen(xi,x,0.25) # h=1/4
p3=parzen(xi,x,1/16) # h=1/16

#plt.plot(x1,y1,'y.'); #画图
plt.plot(x,p1,'r',label='h=1')
plt.plot(x,p2,'b',label='h=1/4')
plt.plot(x,p3,'g',label='h=1/16')
plt.title("hat P(x)关于x的图像")
plt.legend(loc='lower right')
plt.savefig('./2_2.png')