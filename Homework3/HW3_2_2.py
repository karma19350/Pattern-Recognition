# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:50:02 2019

@author: Qingyang Zhong
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as st

num1 = 100
mu = 0
sigma = 1

m=[]
std=[]
s=[]
for i in range(3):
    s1 = np.random.normal(mu, sigma, num1)
    m1= np.mean(s1)
    std1= np.std(s1,ddof=0)
    m.append(m1)
    std.append(std1)
    s.append(s1)

for i in range(3):
    print("mean:%f" %m[i])
    print("std:%f" %std[i])

s_fit = np.linspace(-3.5, 3.5,num=200)
plt.plot(s_fit, st.norm(mu, sigma).pdf(s_fit), lw=2, c='r',label=u'标准正态分布')
plt.plot(s_fit, st.norm(m[0], std[0]).pdf(s_fit), lw=2, c='b',label='sample1')
plt.plot(s_fit, st.norm(m[1], std[1]).pdf(s_fit), lw=2, c='g',label='sample2')
plt.plot(s_fit, st.norm(m[2], std[2]).pdf(s_fit), lw=2, c='y',label='sample3')
plt.legend(loc='lower right')
plt.rcParams['axes.unicode_minus'] = False 
plt.savefig('./normal_100.png')