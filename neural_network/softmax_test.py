#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:22:38 2018

@author: jarvis
"""

import numpy as np;
a = np.random.rand(5);
print(a)
exp_a = np.exp(a);
print(exp_a)

answer = exp_a/exp_a.sum();
print(answer)
print(answer.sum())

A = np.random.rand(100,5);
print(A)

exp_A = np.exp(A);
answer_A = exp_A/exp_A.sum(axis=1,keepdims=True);

print(exp_A.sum(axis=1,keepdims=True))