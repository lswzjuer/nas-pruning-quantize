# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-12-13 15:22:05
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-12-29 21:05:16

import numpy as np 
import os 
import matplotlib.pyplot as plt
import pickle


plt.rcParams['font.sans-serif']=['FangSong'] 



xdata1 = [1,2,3,4,8,32]
ydata1 = [78.62,88.59,89.95,90.27,90.53,92.35]
ydata2 = [31.34,35.20,30.60,28.43,27.01,26.68]
ydata3 = [round(ydata1[i]-ydata2[i],2) for i in range(len(ydata1))]


colorlist=["black","lightcoral","orange","chocolate","gold","green","blue","red"]
fig = plt.figure()
sub1 = fig.add_subplot(1, 1, 1)
# sub1.set_title("BNN")
sub1.plot(xdata1,ydata1,color = "green",  linestyle = '-',marker='*',label=u"攻击前准确率")
sub1.plot(xdata1,ydata2,color = "red",  linestyle = '-',marker='v',label=u"攻击后准确率")


my_x_ticks = [1,2,3,4,8,32]
plt.xticks(my_x_ticks)
# sub1.set_xlim(0, 7)
# sub1.set_ylim(50, 100)  
sub1.set_xlabel(u'量化位宽(bit)',fontdict={'weight': 'normal', 'size': 14})
sub1.set_ylabel(u'准确率(%)',fontdict={'weight': 'normal', 'size': 14})
sub1.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度 
sub1.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
sub1.grid(linestyle='-.')
plt.legend(bbox_to_anchor=[1,0.6])

# 画右轴
ax2 = sub1.twinx() # this is the important function
ax2.plot(xdata1,ydata3,color = "blue",  linestyle = '-',marker='o',label=u"准确率损失")
for a, b in zip(xdata1, ydata3):
    plt.text(a, b, b,va="bottom",ha="center")


ax2.set_ylim(10,80) 
ax2.set_ylabel('准确率损失(%)',fontdict={'weight': 'normal', 'size': 14})
plt.legend(bbox_to_anchor=[1,0.45])
# plt.show()
plt.savefig("./attack_loss.png")