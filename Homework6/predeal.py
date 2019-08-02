# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:43:41 2019

@author: wangpeng884112
"""
import scipy.io as scio 
dataFile = 'Sogou_webpage.mat'  
data = scio.loadmat(dataFile)  
feature = data['wordMat'] 
label = data['doclabel']
label = label[:,0]