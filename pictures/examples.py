# -*- coding:utf-8 -*-
import numpy as np  
import matplotlib.pyplot as plt
import  matplotlib

font = {
    'family' : 'SimHei'
}
matplotlib.rc('font', **font)

["006539","NFYXJZHHC","南方优选价值混合C","混合型","NANFANGYOUXUANJIAZHIHUNHEC"]
["100032","FGZZHLZSZQ","富国中证红利指数增强","股票指数","FUGUOZHONGZHENGHONGLIZHISHUZENGQIANG"]
["003318","JSZZ500HYZXDBD","景顺中证500行业中性低波动","股票指数","JINGSHUNZHONGZHENG500HANGYEZHONGXINGDIBODONG"]
["001986","QHKYRGZNZTHH","前海开源人工智能主题混合","混合型","QIANHAIKAIYUANRENGONGZHINENGZHUTIHUNHE"]
["004744","YFDCYBETFLJC","易方达创业板ETF联接C","联接基金","YIFANGDACHUANGYEBANETFLIANJIEC"]
["165312","JXYS50","建信央视50","股票指数","JIANXINYANGSHI50"]
i=1
index=321
subIndex=321
def calc(value):
    global index
    day="6-1,6-2,6-3".split(',')
    #v="建信50 | 0.8502/0.8499/0.8496    | 0.8263/0.8239/0.8253 | 0.0260/0.0246".split('|')
    v=value.split('|')
    name=v[0]
    myValue=[]
    for m in v[1].split('/'):
        myValue.append(float(m))
    curValue=[]
    for m in v[2].split('/'):
        curValue.append(float(m))
    z=[]
    #整理差值数据
    for i in range(len(day)):
        z.append((myValue[i]-curValue[i])*10)
    
    plt.figure(1)
    plt.subplot(index)
    plt.plot(day, myValue, color = 'blue', linewidth = 2.0, linestyle = '-',label="持仓成本价")
    plt.plot(day, curValue, color = 'red', linewidth = 2.0, linestyle = '--',label="当前净值")
    plt.legend()  #显示上面的label
    plt.title(name) #添加标题

    plt.figure(2)
    plt.subplot(index)
    plt.plot(day, z, color = 'red', linewidth = 2.0, linestyle = '-',label="差值")
    plt.legend()  #显示上面的label
    plt.title(name+"差值") #添加标题
    index=index+1
    return z

def sub(name,value):
    plt.figure(1)
    plt.subplot(index)
    index=index+1

    plt.plot(day, value, color = 'red', linewidth = 2.0, linestyle = '-',label="差值")
    plt.legend()  #显示上面的label
    plt.title(name+"差值") #添加标题
    
with open("./a.txt",'r', encoding='UTF-8') as file:
    for line in file:
        z=calc(line)
        name=(line.split('|'))[0]
        #print(name)
plt.show()