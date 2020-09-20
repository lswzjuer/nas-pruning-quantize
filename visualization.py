# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-18 01:04:24
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-20 20:07:33

import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter,OrderedDict

from utils import getParams,getFlops,get_model_complexity_info,get_logger
from models import get_models
import os 
from datasets.classification import getDataloader
import torch
import math

Models=["EfficientNetB0","GoogLeNetV1","MobileNetV1","MobileNetV2",
        'PreActResNet18','PreActResNet34','PreActResNet50','PreActResNet101',
        'ResNet18','ResNet34','ResNet50','ResNet101',
        'resnet20','resnet32','resnet44','resnet56','resnet110',
        'SENet18','ShuffleNetG2','ShuffleNetG3',
        'ShuffleNetV2_0_5','ShuffleNetV2_1','ShuffleNetV2_1_5','ShuffleNetV2_2',
        'VGG11','VGG13','VGG16','VGG19']

# Count the calculations and parameters of common models
def paramsFlopsCounter(models,num_classes=10,input_shape=(3,32,32)):
    logger=get_logger("./")
    for modelname in models:
        model=get_models(modelname,num_classes=10)
        pa1=getParams(model)
        fl1=getFlops(model,input_shape)
        fl2,pa2=get_model_complexity_info(model,input_shape)
        #logger.info("{}  v1: {}--{} ".format(model,pa1,fl1))
        logger.info("{}  v1: {}--{}  v2: {}--{}".format(modelname,pa1,fl1,pa2,fl2))


class weightActivateCollect(object):
    """docstring for weightActivateCollect"""
    def __init__(self, imagenum,dataloader,modelname,numclasses,checkpoint):
        super(weightActivateCollect, self).__init__()
        self.num=imagenum
        self.dataloader=dataloader
        self.modelname=modelname
        self.checkpoint=checkpoint
        self.numclasses=numclasses
        self.weightops=['Conv2d',"BatchNorm2d"]
        self.activaops=['Conv2d',"BatchNorm2d","ReLU"]
        self.loadModelAndCheckpoint()


    def loadModelAndCheckpoint(self):
        self.model=get_models(self.modelname,num_classes=self.numclasses)
        assert os.path.isfile(self.checkpoint)
        checkpoint = torch.load(self.checkpoint, map_location=torch.device("cpu"))
        extras=checkpoint["extras"]
        epoch=checkpoint["epoch"]
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()


    def weightactCollect(self):

        weightdict=OrderedDict()
        actidict=OrderedDict()
        nameList=[]
        outputs=[]

        def _hook(self,input,output):
            outputs.append(output.cpu().detach().numpy().astype(np.float32))

        # collect weight and register activations hooks
        for name,module in self.model.named_modules():
            if type(module).__name__ in self.weightops:
                weightname=name+".weight"
                weightdict[weightname]={"m":module.weight.cpu().detach().numpy().astype(np.float32),"type":type(module).__name__}
                print(name,type(module).__name__,module.weight.size())
            if type(module).__name__ in self.activaops:
                nameList.append((name,type(module).__name__))
                module.register_forward_hook(_hook)
                print(name,type(module).__name__)

        # forward
        self.model.to(torch.device("cpu"))
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                inputs = inputs.to(torch.device("cpu"))
                _ = self.model(inputs)
                break
        assert len(nameList)==len(outputs)
        for i in range(len(nameList)):
            actidict[nameList[i][0]]={"m": outputs[i],"type":nameList[i][1]}
        return weightdict,actidict


    def weighthistgram(self,weightdict,path):
        assert len(weightdict)!=0,"the weightdict is empty!"
        if not os.path.exists(path):
            os.mkdir(path)
        modelfolder=os.path.join(path,self.modelname)
        if not os.path.exists(modelfolder):
            os.mkdir(modelfolder)
        # total histgram
        for name,values in weightdict.items():
            # conv o,i,k,k
            if len(values["m"].shape)==4:
                histgram(name,values["m"],os.path.join(modelfolder,"total"),
                    index=None,type=values["type"],sqnr=SQNRComputer(values["m"],weight=True))
            else:
                histgram(name,values["m"],os.path.join(modelfolder,"total"),
                    index=None,type=values["type"])


        # # sub creat
        # for name,values in weightdict.items():
        #     # conv o,i,k,k
        #     if len(values["m"].shape)==4:
        #         for i in range(len(values["m"])):
        #             histgram(name,values["m"][i],os.path.join(modelfolder,name),
        #                 index=i,type=values["type"],sqnr=SQNRComputer(values["m"][i],weight=False))
        #     elif len(values["m"].shape)==1:
        #             histgram(name,values["m"],os.path.join(modelfolder,name),
        #                 index=None,type=values["type"],sqnr=SQNRComputer(values["m"][i],weight=False))


    def actihistgram(self,actdict,path):
        assert len(actdict)!=0,"the weightdict is empty!"
        if not os.path.exists(path):
            os.mkdir(path)
        modelfolder=os.path.join(path,self.modelname)
        if not os.path.exists(modelfolder):
            os.mkdir(modelfolder)
        # total histgram
        for name,values in actdict.items():
            # conv o,i,k,k
            values_=values["m"].mean(axis=0)
            assert len(values_.shape)==3
            histgram(name,values_,os.path.join(modelfolder,"total_act"),
                index=None,type=values["type"],sqnr=SQNRComputer(values_,weight=False))


    def wbinaryhistgram(self,weightdict,path):
        assert len(weightdict)!=0,"the weightdict is empty!"
        if not os.path.exists(path):
            os.mkdir(path)
        modelfolder=os.path.join(path,self.modelname)
        if not os.path.exists(modelfolder):
            os.mkdir(modelfolder)
        # total histgram
        newdict={}
        for name,values in weightdict.items():
            # conv o,i,k,k
            if len(values["m"].shape)==4:
                values["m"]=wbinaryComputer(values["m"])
            newdict[name]=values

        for name,values in newdict.items():
            # conv o,i,k,k
            histgram(name,values["m"],os.path.join(modelfolder,"total_b"),
                index=None,type=values["type"])
        # # sub creat
        # for name,values in newdict.items():
        #     # conv o,i,k,k
        #     if len(values["m"].shape)==4:
        #         for i in range(len(values["m"])):
        #             histgram(name,values["m"][i],os.path.join(modelfolder,name+"_b"),index=i,type=values["type"])
        #     elif len(values["m"].shape)==1:
        #             histgram(name,values["m"],os.path.join(modelfolder,name+"_b"),index=None,type=values["type"])


    def abinaryhistgram(self,actdict,path):
        assert len(actdict)!=0,"the weightdict is empty!"
        if not os.path.exists(path):
            os.mkdir(path)
        modelfolder=os.path.join(path,self.modelname)
        if not os.path.exists(modelfolder):
            os.mkdir(modelfolder)
        # total histgram
        for name,values in actdict.items():
            # conv o,i,k,k
            values_=values["m"].mean(axis=0)
            assert len(values_.shape)==3
            histgram(name,abinaryComputer(values_),os.path.join(modelfolder,"total_act_b"),index=None,type=values["type"])



def histgram(name,values,path,index=None,type=None,sqnr=None):
    if not os.path.exists(path):
        os.mkdir(path)
    picname = name + "_{}".format(index) if index is not None else name
    picnamepath=os.path.join(path,picname+".png")
    # 
    values=values.flatten()
    # from collections import Counter
    # print(Counter(list(values)))
    # set the plt
    fig=plt.figure()
    plt.grid()
    tname=name + " ({})".format(type) if type is not None else name
    stname= tname + "{}(db)".format(sqnr) if sqnr is not None else tname
    plt.title(stname)
    plt.xlabel('value')
    plt.ylabel('frequency')
    #plt.ylim(0,4)
    #plt.xlim(-5,5)
    plt.hist(values,bins="auto",density=True, histtype='bar', facecolor='blue')
    fig.savefig(picnamepath, bbox_inches='tight')
    plt.close(fig)


def safeSign(data):
    bw=np.sign(data)
    bw[bw==0]=1
    return bw

def wbinaryComputer(data):
    '''
    o i k k
    '''
    alpha=np.abs(data).mean(axis=(1,2,3),keepdims=True)
    bw=safeSign(data)
    return bw*alpha

def abinaryComputer(data):
    '''
    c h w
    '''
    alpha=np.abs(data).mean(axis=(0,1,2))
    bw=safeSign(data)
    return bw*alpha 


def SQNRComputer(data,weight=True):
    if weight:
        binary=wbinaryComputer(data)
    else:
        binary=abinaryComputer(data)
    eps=1e-5
    a=np.sum(np.power(data,2))
    b=np.sum(np.power(data-binary,2))+eps
    sqnr=10*np.log10(a/b)
    return str(sqnr)



if __name__ == '__main__':
    paramsFlopsCounter(Models,10,(3,64,64))
    # checkpoint=r"./checkpoints/vgg13_cifar10_300_128_0.007_1/ckpts/vgg13_best.pth.tar"
    # save_path=r"./histogram"
    # cifar10=getDataloader("cifar10","val",64,
    #                     2,True,False)
    # collect=weightActivateCollect(1,cifar10,"VGG13",10,checkpoint)
    # weightdict,actdict=collect.weightactCollect()
    # # histgram
    # collect.weighthistgram(weightdict,save_path)
    # collect.actihistgram(actdict,save_path)
    # collect.wbinaryhistgram(weightdict,save_path)
    # collect.abinaryhistgram(actdict,save_path)

