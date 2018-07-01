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



W1 = np.random.rand(D,M);
b1 = np.random.rand(M);
W2 = np.random.rand(M,K);
b2 = np.random.rand(K);

def forward_prop(X,W1,b1,W2,b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1));
    A = Z.dot(W2) + b2;
    exp_A = np.exp(A);
    Y_Pred = exp_A/exp_A.sum(axis=1,keepdims=True);
    return Y_Pred;

def classification_rate(Y,P):
    no_of_tests = len(Y);
    no_of_correct = 0;
    
    for i in range(len(Y)):
        if(Y[i] == P[i]):
            no_of_correct += 1;
            
    return float(no_of_correct/no_of_tests);
    
Y_Pred = forward_prop(X,W1,b1,W2,b2);   
print(Y_Pred) 
P_Pred = np.argmax(Y_Pred,axis=1);

print(P_Pred)

print('Classification Rate : '  ,  classification_rate(Y,P_Pred));    