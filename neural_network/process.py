#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:32:16 2018

@author: jarvis
"""

import numpy as np;
import pandas as pd;
import pprint;

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix();
    
    #Extract train data from dataset
    X = data[:, :-1];
    Y = data[:, -1];
    
    #Normalize  X columns 2 and 3    
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std();
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std();
    
    N,D = np.shape(X);
    #adding 3 more colmns because we know the category time_of_day has 4 categories
    X2 = np.zeros((N,D+3));
    X2[:, :-4] = X[:, :-1]    
    #One-hot encoding for time__od_day category
#    for n in range(N):
#        t = int(X[n,D-1]);
        #add t index to D and make it as 1
#        X2[n,D + t - 1] = 1;
    
    #one hot encoding method 2
    Z = np.zeros((N,4));
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1;
    X2[:,-4:] = Z;    
#    pprint.pprint(X2);
    return X2,Y;

def get_binary_data():
    X,Y = get_data();
    X2 = X[Y <= 1 ];
    Y2 = Y[Y <= 1 ];
    pprint.pprint(np.shape(Y2));
    
    return X2,Y2;
    
get_binary_data();    