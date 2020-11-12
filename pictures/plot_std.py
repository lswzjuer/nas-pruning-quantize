# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-04 18:04:27
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-12 17:10:35

import numpy as np 
import matplotlib.pyplot as plt



# check model
X_test = np.linspace(-1., 1., num_samples * 2).reshape(-1, 1)
y_test = f_fun(X_test)
y_preds = [model(X_test) for _ in range(300)]
y_preds = np.concatenate(y_preds, axis=1)

plt.plot(X, y, 'b*', label='Training Points')
plt.plot(X_test, np.mean(y_preds, axis=1), 'r-', label='Predict Line')
plt.fill_between(X_test.reshape(-1), np.percentile(y_preds, 2.5, axis=1), np.percentile(y_preds, 97.5, axis=1), color='r', alpha=0.3, label='95% Confidence')
plt.grid()
plt.legend()