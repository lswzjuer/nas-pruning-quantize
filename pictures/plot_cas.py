# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-12-11 23:05:22
# @Last Modified by:   liusongwei
# @Last Modified time: 2021-01-03 22:03:27


# -*- coding:utf-8 -*-
import numpy as np  
import matplotlib.pyplot as plt
import  matplotlib
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif']=['FangSong'] 


x_data = [0,0.02,0.04,0.06,0.08,0.1,0.14,0.18,0.2]

y_data1 = [92.0,91.4,92.2,91.9,90.9,90.1,88.5,85.7,84.5]

y_data2 = [2.87,2.89,2.72,2.70,2.64,2.38,2.26,2.05,1.75]

y_data22 = [ x-0.77 for x in y_data2]


fig = plt.figure()
ax1 = fig.add_subplot(111)

# y_data2作为左轴  y_data1右轴
ax1.scatter(x_data,y_data22,marker='o',color = "blue",label="参数量"+"$-\lambda$",)
# 在画出拟合的曲线
parameter = np.polyfit(x_data, y_data22, 2) 
f = np.poly1d(parameter)
ax1.plot(x_data,f(x_data),label="参数量"+"$-\lambda$"+u" 拟合曲线")


my_x_ticks = np.arange(0,0.22,0.02)
#my_y_ticks = np.arange(0, 100, 5)
plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks)
ax1.set_xlabel('$\lambda(\\times 10^{-2})$',fontdict={'weight': 'normal', 'size': 14})
ax1.set_ylabel('参数量(M)',fontdict={'weight': 'normal', 'size': 14})
ax1.set_xlim(0, 0.2)
ax1.set_ylim(1, 2.5)   
# ax1.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度 
# ax1.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
ax1.grid(linestyle='-.')
# # reverse the order
ax1.legend(loc=4)

# 画右轴
ax2 = ax1.twinx() # this is the important function
ax2.scatter(x_data,y_data1,marker='*',color = "red",label="准确率"+"$-\lambda$")
parameter = np.polyfit(x_data, y_data1, 2) 
f = np.poly1d(parameter)
ax2.plot(x_data,f(x_data),color="red", label="准确率"+"$-\lambda$"+u" 拟合曲线")

ax2.set_ylim(81.5, 93) 
ax2.set_ylabel(u'准确率(%)',fontdict={'weight': 'normal', 'size': 14})

plt.legend(loc=1)
# plt.show()
plt.savefig("./cas.png")