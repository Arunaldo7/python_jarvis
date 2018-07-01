#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:41:47 2018

@author: jarvis
"""

import numpy as np;
import matplotlib.pyplot as plt;

Nclass = 500;
D = 2;
M = 3;
K = 3;

#form 3 sets of X Inputs
X1 = np.random.rand(Nclass,D) + np.array([0,-2]);
X2 = np.random.rand(Nclass,D) + np.array([2,2]);
X3 = np.random.rand(Nclass,D) + np.array([-2,-2]);

#Stack inputs to form one matrix of inputs
X = np.vstack([X1,X2,X3])

print(X)

#form Outputs - Row Matrix , number of column dimensions same as rows of X
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
print(Y)

plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=0.5)
plt.show();



W1 = np.random.rand(D,M)
b1 = np.rand