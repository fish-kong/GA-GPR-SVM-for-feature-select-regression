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

def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return idx,pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)#交叉操作
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):#变异操作
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

def fit(pop,ptr,ttr,pte,tte):
    #适应度函数
    # svr与elm gpr均采用默认参数
    f_value_svm=[]
    f_value_elm=[]
    f_value_gpr=[]
    for i in range(pop.shape[0]):
        if np.sum(pop[i])==0:#如果染色体i全为0，则重新生成一次第i个染色体
            pop[i]=np.random.randint(2, size=(1, DNA_SIZE))   
        p_tr,p_te=translateDNA(pop[i],ptr,pte)
        # svr
        clf = svm.SVR() 
        clf.fit(p_tr,ttr)
        result_svm = clf.predict(p_te)
        #elm
        Weights,biases,w_out,TF,TYPE=elm.fit(p_tr,ttr)
        result_elm=elm.predict(p_te,Weights,biases,w_out,TF,TYPE)
        #gpr
        reg = GaussianProcessRegressor()
        reg.fit(p_tr,ttr)#这是拟合高斯过程回归的步骤
        result_gpr= reg.predict(p_te)
        
        f_value_svm.append(get_error(tte,result_svm))
        f_value_elm.append(get_error(tte,result_elm))        
        f_value_gpr.append(get_error(tte,result_gpr))

    return np.array(f_value_svm)+np.array(f_value_elm)+np.array(f_value_gpr)  #以svr与gpr的误差和作为适应度值 目的是找到一组特征 使得两者的输出误差同时最小
# In[2]
# load data
m=np.array(pd.read_excel('sampleWindspeed.xlsx',header=None))

#data=m[0:485,0];n=386  # 数据集1
data=m[0:675,1];n=539 # 数据集2
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

# In[3]
# default parameter 
DNA_SIZE = ptr.shape[1] # DNA length
POP_SIZE = 20           # population size
CROSS_RATE = 0.5        # mating probability (DNA crossover)
MUTATION_RATE = 0.5     # mutation probability
N_GENERATIONS = 100     # iteration size
# initialize the pop DNA
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   
best_pop=[]
best_fitness=1000000000000
trace=np.zeros([N_GENERATIONS,1])

for iter in range(N_GENERATIONS):
    print('进化至第',iter+1,'代')
    fitness = fit(pop,ptr,ttr,pte,tte)    # compute function value by extracting DNA
    # GA part (evolution)
    loc_pop=pop[np.argmin(fitness)]
    loc_fitness=np.min(fitness)
    
    if loc_fitness<best_fitness:
        best_fitness=loc_fitness
        best_pop=loc_pop
    trace[iter]=best_fitness

    idx,pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

# In[4]
# result analysis
plt.figure()
plt.plot(trace)
plt.xlabel('iter')
plt.ylabel('best_fitness')
plt.show()

n=np.array(np.where(best_pop))
print('--------------------------------------')
print('原本一共有',ptr.shape[1],'个特征')
print('经ga+svm+gpr算法筛选出',best_pop.sum(),'个特征')
print('最优的特征组合为：',n[0]+1)
# python数组索引是从0开始编号的，因此我们对结果+1，改成我们习惯的读取方式

# In[5]
# 利用筛选出来的特征重新进行svr与gpr的建模

p_tr,p_te=translateDNA(best_pop,ptr,pte)
# svr
clf=svm.SVR()
clf.fit(p_tr,ttr)
result_svm = clf.predict(p_te)
#gpr
reg=GaussianProcessRegressor()
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
