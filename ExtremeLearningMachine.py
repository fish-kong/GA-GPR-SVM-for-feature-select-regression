# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:37:14 2019

@author: KLY
"""

'极限学习机--extreme learning machine'
import numpy as np


'极限学习机训练函数'
def fit(P,T,N=10,C=100000000000000,TF='sig',TYPE=0):
    # P 输入数据 n*m  n-> samples m->features
    # T 输出数据
    # N 隐含层节点
    # C 正则化参数
    # TF 隐含层激活函数
    # TYPE=1 分类任务  =0 回归任务
    
    n,m=P.shape
    if TYPE == 1:
        y=np.zeros([n,T.max()+1])
        for i in range(n):
            y[i,T[i]]=1
        T=np.copy(y)
    
    '输入权重'
    Weights = 2*np.random.rand(m,N)-1
    '隐含层偏置'
    biases=np.random.rand(1,N)
    
    temp=np.matmul(P,Weights)+np.tile(biases,[n,1])
    if TF=='sig':
        H=1/(1+np.exp(temp))
    '输出权重计算'
    w_out=np.matmul(np.matmul(np.linalg.pinv(np.matmul(H.T,H)+1/C),H.T),T)
    return Weights ,biases ,w_out, TF, TYPE

def predict(P,Weights,biases,w_out,TF,TYPE):
    n,m=P.shape
    temp=np.matmul(P,Weights)+np.tile(biases,[n,1])
    if TF=='sig':
        H=1/(1+np.exp(temp))
    T=np.matmul(H,w_out)
    
    if TYPE==1:
        T_predict=np.argmax(T,axis=1)
    if TYPE==0:
        T_predict=T
    return T_predict













