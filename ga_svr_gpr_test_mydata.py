"""
Visualize Genetic Algorithm to find a maximum point in a function.

Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# In[1]
def fit(pop,ptr,ttr,pte,tte):
    f_value_svm=[]
    f_value_gpr=[]
    for i in range(pop.shape[0]):
        p_tr,p_te=translateDNA(pop[i],ptr,pte)
        # svr
        clf = svm.SVR(kernel='rbf', C=10, gamma=100) 
        clf.fit(p_tr,ttr)
        result_svm = clf.predict(p_te)
        #gpr
        reg = GaussianProcessRegressor(kernel=C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10)))
        reg.fit(p_tr,ttr)#这是拟合高斯过程回归的步骤
        result_gpr= reg.predict(p_te)
        
        f_value_svm.append(get_error(tte,result_svm))
        f_value_gpr.append(get_error(tte,result_gpr))

    return np.array(f_value_svm)+np.array(f_value_gpr)  #以svr与gpr的误差和作为适应度值 目的是找到一组特征 使得两者的输出误差同时最小
def get_error(records_real, records_predict):
    #均方根误差 估计值与真值 偏差
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real))
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
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child
# In[2]
# load data
m = loadmat("data.mat")
ptr=m['P_train'][:,0:10]
ttr=m['T_train']
pte=m['P_test'][:,0:10]
tte=m['T_test']

#==============================================================================
# # 输入数据归一化
# #训练输入归一化
# scale = StandardScaler()
# scale_fit = scale.fit(ptr)
# ptr = scale_fit.transform(ptr)
# #预测时用同一个scale_fit归一化，再预测
# pte = scale_fit.transform(pte)
# #输出数据归一化
# #训练归一化
# scale2 = StandardScaler()
# scale_fit2 = scale2.fit(ttr)
# ttr = scale_fit2.transform(ttr)
# #预测时用同一个scale_fit归一化，再预测
# tte = scale_fit2.transform(tte)
#==============================================================================
# In[3]
# default parameter 
DNA_SIZE = ptr.shape[1] # DNA length
POP_SIZE = 10           # population size
CROSS_RATE = 0.5        # mating probability (DNA crossover)
MUTATION_RATE = 0.5     # mutation probability
N_GENERATIONS = 10     # iteration size
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

    pop = select(pop, fitness)
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
print('经ga+svm算法筛选出',best_pop.sum(),'个特征')
print('最优的特征组合为：',n[0]+1)
# python数组索引是从0开始编号的，因此我们对结果+1，改成我们习惯的读取方式

# In[5]
# 利用筛选出来的特征重新进行svr与gpr的建模

p_tr,p_te=translateDNA(best_pop,ptr,pte)
# svr
clf = svm.SVR(kernel='rbf', C=10, gamma=100) 
clf.fit(p_tr,ttr)
result_svm = clf.predict(p_te)
#gpr
reg = GaussianProcessRegressor(kernel=C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10)))
reg.fit(p_tr,ttr)#这是拟合高斯过程回归的步骤
result_gpr= reg.predict(p_te)
plt.figure()
plt.plot(tte,c='r',label='true')
plt.plot(result_svm,c='g',label='svr')
plt.plot(result_gpr,c='b',label='gpr')
plt.legend()
plt.show()
print('支持向量回归结果的均方根误差为',get_error(tte,result_svm))
print('高斯过程回归结果的均方根误差为',get_error(tte,result_gpr))



