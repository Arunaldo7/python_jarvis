#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 18:28:23 2018

@author: jarvis
"""

import numpy as np;
from process import get_data;

X,Y = get_data();

#set hidden layers, input features,output layers
M = 5;
D = np.shape(X)[1];
#will give unique number of Y values -- which is equal to the number of output layers
K = len(set(Y));

#initialise weights and biases
W1 = np.random.rand(D,M);
B1 = np.zeros(M);

W2 = np.random.rand(M,K);
B2 = np.zeros(K);

def softmax(a):
    #take exp for softmax exp(a) / sum(exp(a))
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=1,keepdims=True);
    
def forward(X,W1,W2,B1,B2):
    #sum of features dot weights and bias
    a = X.dot(W1) + B1;
    #get tanh
    #method 1
    #Z = 1 / 1 + np.exp(-a);
    #method 2
    Z = np.tanh(a);
    
    #sum of Z dot hidden layer weights and bias
    A = Z.dot(W2) + B2;
    
    #get prediction by softmax
    P_Y_given_X = softmax(A);
    
    return P_Y_given_X;

def classification_rate(Y,P):
    return np.mean(Y == P)
    
P_Y_given_X = forward(X,W1,W2,B1,B2);
predictions = np.argmax(P_Y_given_X,axis=1)

print('P_Y_given_X : ' , P_Y_given_X)
print('Score : ' , classification_rate(Y,predictions))