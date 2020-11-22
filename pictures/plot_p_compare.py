# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-11-19 20:43:55
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-19 23:09:38



import numpy as np 
import os 
import matplotlib.pyplot as plt
import pickle





xdata1 = [1,2,2.5,3,3.5,4,5,6]
ydata1 = [74.82,82.53,83.30,83.47,83.32,83.20,82.17,60.32]


xdata2 = [1,2,2.5,3,3.5,4,5,6]
ydata2 = [79.65,81.17,82.95,83.06,82.10,76.69,65.4,58.2]



# xdata1 = [1,2,3,4,5,6]
# ydata1 = [74.82,82.53,83.47,83.20,82.17,60.32]


# xdata2 = [1,2,3,4,5,6]
# ydata2 = [79.65,81.17,83.06,76.69,65.4,58.2]




colorlist=["black","lightcoral","orange","chocolate","gold","green","blue","red"]
fig = plt.figure()
# plt.margins(0.05)
# plt.subplots_adjust(top=0.15)
# add subplot1
sub1 = fig.add_subplot(1, 2, 1)
sub1.set_title("BNN")
sub1.plot(xdata1,ydata1,color = "green",  linestyle = '-',marker='v')
for a, b in zip(xdata1, ydata1):
    if a ==3:
        plt.text(a, b, b,va="bottom",ha="center")
plt.axhline(y=79.7,c="r",ls="--",lw=2)
my_x_ticks = np.arange(0, 7, 1)
my_y_ticks = np.arange(50, 100, 5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
sub1.set_xlim(0, 7)
sub1.set_ylim(50, 100)  
sub1.set_xlabel('$\\rho$')
sub1.set_ylabel('$Accuracy$')
sub1.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度 
sub1.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
sub1.grid(linestyle='-.')



sub2 = fig.add_subplot(1, 2, 2)
sub2.set_title("Bireal")
sub2.plot(xdata2,ydata2,color = "green",  linestyle = '-',marker='v')
for a, b in zip(xdata2, ydata2):
    if a ==3:
        plt.text(a, b, b,va="bottom",ha="center")
plt.axhline(y=81.27,c="r",ls="--",lw=2)
my_x_ticks = np.arange(0, 7, 1)
my_y_ticks = np.arange(50, 100, 5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
sub2.set_xlim(0, 7)
sub2.set_ylim(50, 100)  
sub2.set_xlabel('$\\rho$')
# sub1.set_ylabel('$Accuracy$')
sub2.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度 
sub2.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
sub2.grid(linestyle='-.')

# plt.show()
# plt.close()
plt.savefig("./p_compare.png")


