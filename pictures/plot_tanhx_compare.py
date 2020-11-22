# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-11-18 20:49:57
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-19 22:38:00
import numpy as np 
import os 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def smoothValue(olddata,factor):
    last = olddata[0]
    smoothed = []
    for point in olddata:
        smoothed_val = last * factor + (1 - factor) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def readfile(path):
    rd = pd.read_csv(path)
    df = pd.DataFrame(rd)
    data = df.loc[:,"Value"].values
    data = smoothValue(data,0.7)
    return data



if __name__=="__main__":
    folder=r"./tanh3x_compare/resnet20"

    resnet20_float = readfile(os.path.join(folder,"resnet20.csv"))
    resnet20_dorefa = readfile(os.path.join(folder,"resnet20-doreafa1w1a.csv"))

    bnn_original = readfile(os.path.join(folder,"resnet20-bnn-original.csv"))
    bnn_tanh3x = readfile(os.path.join(folder,"resnet20-bnn-tanh3x.csv"))

    xnor_original = readfile(os.path.join(folder,"resnet20-xnor-original.csv"))
    xnor_tanh3x = readfile(os.path.join(folder,"resnet20-xnor-tanh3x-bnn.csv"))
    
    bireal_original = readfile(os.path.join(folder,"resnet20-bireal-original.csv"))
    bireal_tanh3x = readfile(os.path.join(folder,"resnet20-bireal-tanh3x.csv"))


    folder=r"./tanh3x_compare/vgg"

    vggsmall_float = readfile(os.path.join(folder,"vggsmall.csv"))
    vggsmall_dorefa = readfile(os.path.join(folder,"vggsmall-dorefa1w1a.csv"))

    vbnn_original = readfile(os.path.join(folder,"vggsmall-xnor-original.csv"))
    vbnn_tanh3x = readfile(os.path.join(folder,"vggsmall-xnor-tanh3x.csv"))

    vxnor_original = readfile(os.path.join(folder,"vggsnall-bnn-original.csv"))
    vxnor_tanh3x = readfile(os.path.join(folder,"vggsnall-bnn-tanh3x.csv"))
    


    colorlist=["black","lightcoral","orange","chocolate","gold","green","blue","red"]
    fig = plt.figure()
    # plt.margins(0.05)
    # plt.subplots_adjust(top=0.15)
    # add subplot1
    sub1 = fig.add_subplot(1,1,1)
    sub1.set_title("ResNet-20")
    # sub1.plot(range(len(resnet20_float)),resnet20_float,color = "black",  linestyle = '-.',
    #           label=r"$float$")
    sub1.plot(range(len(resnet20_dorefa)),resnet20_dorefa,color ="black",  linestyle = ':',
              label=r"$Dorefa$")

    sub1.plot(range(len(bnn_original)),bnn_original,color = "green", linestyle = ':',
              label=r"$BNN$")
    sub1.plot(range(len(bnn_tanh3x)),bnn_tanh3x,color = "green", linestyle = '-',
              label=r"$BNN(tanh)$")

    sub1.plot(range(len(xnor_original)),xnor_original,color = "red", linestyle = ':',
              label=r"$XNOR$")
    sub1.plot(range(len(xnor_tanh3x)),xnor_tanh3x,color = "red", linestyle = '-',
              label=r"$XNOR(tanh)$")

    sub1.plot(range(len(bireal_original)),bireal_original,color = "blue", linestyle = ':',
              label=r"$Bireal$")
    sub1.plot(range(len(bireal_tanh3x)),bireal_tanh3x,color = "blue", linestyle = '-',
              label=r"$Bireal(tanh)$")


    # axins = sub1.inset_axes((0.1, 0.1, 0.3, 0.3))

    # axins = inset_axes(sub1, width="40%", height="30%",loc='lower left',
    #                    bbox_to_anchor=(0.1, 0.1, 1, 1),
    #                    bbox_transform=sub1.transAxes)
    axins = sub1.inset_axes((0.3, 0.2, 0.3, 0.3))
    #在子坐标系中绘制原始数据
    # sub1.plot(range(len(resnet20_float)),resnet20_float,color = "black",  linestyle = '-.',
    #           label=r"$float$")
    axins.plot(range(len(resnet20_dorefa)),resnet20_dorefa,color ="black",  linestyle = ':',
              label=r"$Dorefa$")

    axins.plot(range(len(bnn_original)),bnn_original,color = "green", linestyle = ':',
              label=r"$BNN$")
    axins.plot(range(len(bnn_tanh3x)),bnn_tanh3x,color = "green", linestyle = '-',
              label=r"$BNN(tanh)$")

    axins.plot(range(len(xnor_original)),xnor_original,color = "red", linestyle = ':',
              label=r"$XNOR$")
    axins.plot(range(len(xnor_tanh3x)),xnor_tanh3x,color = "red", linestyle = '-',
              label=r"$XNOR(tanh)$")

    axins.plot(range(len(bireal_original)),bireal_original,color = "blue", linestyle = ':',
              label=r"$Bireal$")
    axins.plot(range(len(bireal_tanh3x)),bireal_tanh3x,color = "blue", linestyle = '-',
              label=r"$Bireal(tanh)$")
    # axins.set_xlim(250, 300)
    # axins.set_ylim(ylim0, ylim1)
    # 设置放大区间
    zone_left = 260
    zone_right = 290

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.5 # x轴显示范围的扩展比例
    y_ratio = 0.1 # y轴显示范围的扩展比例
    x=range(len(resnet20_dorefa))
    # X轴的显示范围
    xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((resnet20_dorefa[zone_left:zone_right], bnn_original[zone_left:zone_right], 
                    bnn_tanh3x[zone_left:zone_right],  xnor_original[zone_left:zone_right],
                     xnor_tanh3x[zone_left:zone_right], bireal_original[zone_left:zone_right],
                     bireal_tanh3x[zone_left:zone_right],))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    mark_inset(sub1, axins, loc1=2, loc2=4)

    # my_x_ticks = np.arange(0, 300, 10)
    # # my_y_ticks = np.arange(0, 100, 5)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    sub1.set_xlabel('Epoch')
    sub1.set_ylabel('$Accuracy$')
    # sub1.set_xlim(10, 305)
    # sub1.set_ylim(40, 92)   
    sub1.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度 
    sub1.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
    sub1.grid(linestyle='-.')
    # reverse the order
    sub1.legend(loc=4)
    plt.savefig("./tanh3x_compare1.png")
    # ax1=plt.gca()##获取坐标轴信息,gca=get current axic
    # ax1.xaxis.set_ticks_position('bottom')##位置有bottom(left),top(right),both,default,none
    # ax1.yaxis.set_ticks_position('left')##定义坐标轴是哪个轴，默认为bottom(left)
    # ax1.spines['bottom'].set_position(('data',50 ))##移动x轴，到y=0
    # # ax.spines['left'].set_position(('data',-0.5))##还有outward（向外移动），axes（比例移动，后接小数）

    fig = plt.figure()
    sub2 = fig.add_subplot(1, 1, 1)
    sub2.set_title("Vggsmall")
    # sub2.plot(range(len(vggsmall_float)),vggsmall_float,color = "black",  linestyle = '-.',
    #           label=r"$float$")
    sub2.plot(range(len(vggsmall_dorefa)),vggsmall_dorefa,color ="black",  linestyle = ':',
              label=r"$Dorefa$")

    sub2.plot(range(len(vbnn_original)),vbnn_original,color = "green", linestyle = ':',
              label=r"$BNN$")
    sub2.plot(range(len(vbnn_tanh3x)),vbnn_tanh3x,color = "green", linestyle = '-',
              label=r"$BNN(tanh)$")

    sub2.plot(range(len(vxnor_original)),vxnor_original,color = "red", linestyle = ':',
              label=r"$XNOR$")
    sub2.plot(range(len(vxnor_tanh3x)),vxnor_tanh3x,color = "red", linestyle = '-',
              label=r"$XNOR(tanh)$")

    axins2 = sub2.inset_axes((0.3, 0.2, 0.3, 0.3))
    #在子坐标系中绘制原始数据
    # sub1.plot(range(len(resnet20_float)),resnet20_float,color = "black",  linestyle = '-.',
    #           label=r"$float$")
    axins2.plot(range(len(vggsmall_dorefa)),vggsmall_dorefa,color ="black",  linestyle = ':',
              label=r"$Dorefa$")

    axins2.plot(range(len(vbnn_original)),vbnn_original,color = "green", linestyle = ':',
              label=r"$BNN$")
    axins2.plot(range(len(vbnn_tanh3x)),vbnn_tanh3x,color = "green", linestyle = '-',
              label=r"$BNN(tanh)$")

    axins2.plot(range(len(vxnor_original))[:300],vxnor_original[:300],color = "red", linestyle = ':',
              label=r"$XNOR$")
    axins2.plot(range(len(vxnor_tanh3x)),vxnor_tanh3x,color = "red", linestyle = '-',
              label=r"$XNOR(tanh)$")

    # axins.set_xlim(250, 300)
    # axins.set_ylim(ylim0, ylim1)
    # 设置放大区间
    zone_left = 260
    zone_right = 290

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.5 # x轴显示范围的扩展比例
    y_ratio = 0.1 # y轴显示范围的扩展比例
    x=range(len(resnet20_dorefa))
    # X轴的显示范围
    xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((vggsmall_dorefa[zone_left:zone_right], vbnn_original[zone_left:zone_right], 
                    vbnn_tanh3x[zone_left:zone_right],  vxnor_original[zone_left:zone_right],
                     vxnor_tanh3x[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 调整子坐标系的显示范围
    axins2.set_xlim(xlim0, xlim1)
    axins2.set_ylim(ylim0, ylim1)
    mark_inset(sub2, axins2, loc1=2, loc2=4)


    # my_x_ticks = np.arange(0, 300, 10)
    # my_y_ticks = np.arange(0, 100, 5)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    # sub2.set_xlim(10, 305)
    # sub2.set_ylim(40, 95)  
    sub2.set_xlabel('Epoch')
    sub2.set_ylabel('$Accuracy$')
    sub2.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度 
    sub2.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
    sub2.grid(linestyle='-.')
    # reverse the order
    sub2.legend(loc=4)
    # plt.show()
    # plt.close()
    # plt.savefig("./tanh3x_comparev2.png")
    plt.savefig("./tanh3x_compare2.png")



