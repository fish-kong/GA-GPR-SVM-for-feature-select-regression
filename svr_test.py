

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:35:40 2019

@author: KLY
"""

from __future__ import division
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
 
rng = np.random.RandomState(0)
 
#############################################################################
# 生成随机数据
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()
 
# 在标签中对每50个结果标签添加噪声
 
y[::50] += 2 * (0.5 - rng.rand(int(X.shape[0]/50)))
 
X_plot = np.linspace(0, 5, 100000)[:, None]
 
#############################################################################
# 训练SVR模型
 
#训练规模
train_size = 100
#初始化SVR
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
#记录训练时间
t0 = time.time()
#训练
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
 
t0 = time.time()
#测试
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0