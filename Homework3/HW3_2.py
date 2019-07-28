# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:59:53 2019

@author: Qingyang Zhong
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as st

num1 = 10
mu = 0
sigma = 1
np.random.seed(1)
s1 = np.random.normal(mu, sigma, num1)
m1= np.mean(s1)
std1= np.std(s1,ddof=0)
print("mean1:%f" %m1)
print("std1:%f" %std1)

num2 = 100
mu = 0
sigma = 1
np.random.seed(1)
s2 = np.random.normal(mu, sigma, num2)
m2= np.mean(s2)
std2= np.std(s2,ddof=0)
print("mean2:%f" %m2)
print("std2:%f" %std2)

num3 = 1000
mu = 0
sigma = 1
np.random.seed(1)
s3 = np.random.normal(mu, sigma, num3)
m3= np.mean(s3)
std3= np.std(s3,ddof=0)
print("mean3:%f" %m3)
print("std3:%f" %std3)

#s = st.norm(mu, sigma).rvs(1000)

#s_fit = np.linspace(s.min(), s.max())
s_fit = np.linspace(-3.5, 3.5)
plt.plot(s_fit, st.norm(mu, sigma).pdf(s_fit), lw=2, c='r',label=u'标准正态分布')
plt.plot(s_fit, st.norm(m1, std1).pdf(s_fit), lw=2, c='b',label='num = 10')
plt.plot(s_fit, st.norm(m2, std2).pdf(s_fit), lw=2, c='g',label='num = 100')
plt.plot(s_fit, st.norm(m3, std3).pdf(s_fit), lw=2, c='y',label='num = 1000')
plt.legend(loc='lower right')
plt.rcParams['axes.unicode_minus'] = False 
plt.savefig('./normal.png')