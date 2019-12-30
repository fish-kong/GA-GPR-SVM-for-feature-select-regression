# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:36:01 2019
@author: KLY
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math
from sklearn.gaussian_process import GaussianProcessRegressor
import ExtremeLearningMachine as elm
import scipy.io as sio
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ

# In[1]
def get_error(records_real, records_predict):
    #均方根误差 估计值与真值 偏差
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real))
def split_data(data,n):
    #n个数据用于训练 剩下的用于测试
    #前10个时刻的数据作为输入  第11个时刻的数据作为输出
    data1=[]
    for i in range(len(data)-11+1):
        data1.append(data[i:(i+11)])
    data1=np.array(data1)
    ptr=data1[0:n,0:10]
    ttr=data1[0:n,10]
    pte=data1[n:,0:10]
    tte=data1[n:,10]
    return ptr,ttr,pte,tte

def translateDNA(pop_i,ptr,pte): 
    # features choose  -假设特征数为10，染色体i为[0 1 1 1 0 1 1 0 1 0],代表选择了第2 3 4 6 7 9个特征
    
    m=np.array(np.where(pop_i==1))
    n=np.sum(pop_i)
    p_tr=np.zeros((ptr.shape[0],n))
    p_te=np.zeros((pte.shape[0],n))
  
    p_tr=ptr[:,m].copy()
    p_te=pte[:,m].copy()
    return  p_tr.reshape((ptr.shape[0],n)),p_te.reshape((pte.shape[0],n))


# In[2]
# load data
m=np.array(pd.read_excel('sampleWindspeed.xlsx',header=None))

data=m[0:485,0];n=386  # 数据集1
#data=m[0:675,1];n=539 # 数据集2
#data=m[0:676,2];n=539 # 数据集3
#data=m[0:964,3];n=770  # 数据集4
plt.figure()
plt.plot(range(n),data[:n],c='b')
plt.plot(range(n,len(data)),data[n:],c='r')
plt.show()


# 数据归一化
data=(data-np.min(data))/(np.max(data)-np.min(data))
ptr,ttr,pte,tte=split_data(data,n)
ptr=np.fliplr(ptr)
pte=np.fliplr(pte)

best_pop=np.array([1,1,1,1,1,1,1,1,1,1])
p_tr,p_te=translateDNA(best_pop,ptr,pte)
# svr
clf=svm.SVR()
clf.fit(p_tr,ttr)
result_svm = clf.predict(p_te)
#gpr
kernel = RQ(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5))
reg=GaussianProcessRegressor(kernel=kernel)
reg.fit(p_tr,ttr)#这是拟合高斯过程回归的步骤
result_gpr= reg.predict(p_te)

#elm
Weights,biases,w_out,TF,TYPE=elm.fit(p_tr,ttr)
result_elm=elm.predict(p_te,Weights,biases,w_out,TF,TYPE)

plt.figure()
plt.plot(tte,c='r',label='true')
plt.plot(result_svm,c='g',label='svr')
plt.plot(result_elm,c='y',label='elm')
plt.plot(result_gpr,c='b',label='gpr')

plt.legend()
plt.show()
print('支持向量回归结果的均方根误差为',get_error(tte,result_svm))
print('极限学习回归结果的均方根误差为',get_error(tte,result_elm))
print('高斯过程回归结果的均方根误差为',get_error(tte,result_gpr))


sio.savemat("data_youhua.mat", {"P_train": p_tr,"T_train": ttr,'P_test':p_te,'T_test':tte})
