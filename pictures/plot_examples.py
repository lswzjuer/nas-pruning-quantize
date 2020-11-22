# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-11-17 20:57:03
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-17 20:57:20


import numpy as np 
import os 
import matplotlib.pyplot as plt
import pickle

AlexNet_acc = [75.43, 75.38, 75.55, 75.37, 75.26, 75.11, 74.30]
ResNet_acc  = [82.48, 82.58, 82.76, 82.65, 82.67, 82.55, 82.40]
names = [0.01, 0.03, 0.05, 0.07, 0.1, 0.5, 1]
x = range(len(names))
plt.plot(x, ResNet_acc,  marker='o', mec='#64B5CD',ms=10,  mfc='#64B5CD', c = '#64B5CD', label='ResNet')

plt.plot(x, AlexNet_acc, marker='^', mec='#C75557',ms=12,  mfc='#C75557', ls='--', c = '#C75557', label='AlexNet')

plt.legend() # 让图例生效
plt.xticks(x, names) # 让x轴的刻度以names标签显示

# 绘制图的数值
for i in range(len(AlexNet_acc)):
    plt.text(x[i], AlexNet_acc[i] + 0.5, '%s' %round(AlexNet_acc[i],3), ha='center', fontsize=10)
for i in range(len(ResNet_acc)):
    plt.text(x[i], ResNet_acc[i] - 1, '%s' %round(ResNet_acc[i],3), ha='center', fontsize=10, va='bottom')

# 调整图与y的边距
plt.margins(0.05)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Hyperparameter C") #X轴标签
plt.ylabel("Accuracy%") #Y轴标签
plt.yticks(np.arange(74, 84, step=1))
# plt.title("Stability Analysis on Hyperparameter C") #标题
# plt.grid(linestyle='--') # 展示网格
plt.grid(False) # 展示网格
ax=plt.gca()##获取坐标轴信息,gca=get current axic
print(ax)
ax.spines['right'].set_color('none')##设置右边框颜色为无
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')##位置有bottom(left),top(right),both,default,none
ax.yaxis.set_ticks_position('left')##定义坐标轴是哪个轴，默认为bottom(left)
ax.spines['bottom'].set_position(('data',74 ))##移动x轴，到y=0
ax.spines['left'].set_position(('data',-0.5))##还有outward（向外移动），axes（比例移动，后接小数）

plt.show()
