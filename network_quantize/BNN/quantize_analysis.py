# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-19 20:57:00
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-16 14:40:10

from collections import OrderedDict
import numpy as np 
import sys
import os 
import torch 
import torch.nn as nn 
import argparse
import time
import yaml
import random
import math
import logging
import torch.backends.cudnn as cudnn
import mobilenetv2_bnn as networka
import resnet18_bnn as networkb
import resnet20_bnn as networkc
import vggsmall_bnn as networkd
from quantize_func import *

sys.path.append("../../")
from  utils import *
from datasets.classification import getDataloader


ARCH_DICT={
    "mobilenetv2": networka,
    "resnet18" : networkb,
    "resnet20" : networkc,
    "vggsmall" : networkd
}


def getArgs():
    parser=argparse.ArgumentParser("Train network in cifar10/100/svhn/mnist/tiny_imagenet")
    # model and train setting
    # parser.add_argument('--arch',default="vggsmall",help="binary resnet models")
    # parser.add_argument('--model',default="vgg_small",help="binary resnet models")

    parser.add_argument('--arch',default="resnet20",help="binary resnet models")
    parser.add_argument('--model',default="resnet20",help="binary resnet models")

    parser.add_argument('--init_type',default="kaiming",help="weight init func")
    # datasets
    parser.add_argument('--datasets',type=str,default="cifar10",help="datset name")
    parser.add_argument('--root',type=str,default="./datasets",help="datset path")
    parser.add_argument('--class_num',type=int,default=10,help="datasets class name")
    parser.add_argument('--flag',type=str,default="train",help="train or eval")
    # lr and train setting
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                            help='number of total epochs to run')
    parser.add_argument('--batch_size',type=int,default=128,help="batch size")
    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay 0 for bireal network')
    parser.add_argument('--workers',type=int,default=2,help="dataloader num_workers")
    parser.add_argument("--pin_memory",type=bool,default=True,help="dataloader cache ")
    parser.add_argument('--cutout',default=False, action='store_true')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume',action='store_true',default=False,help="traing resume")
    parser.add_argument('--resume_path',type=str,default="./checkpoint/model_xx",help="traing resume_path")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--num_best_scores',type=int,default=5,help="num_best_scores")
    parser.add_argument('--optimizer',type=str,default="sgd",choices=["adam","sgd"],help="optimizer")
    parser.add_argument('--scheduler',type=str,default="cos",choices=["cos","step","mstep"],help="scheduler")
    parser.add_argument('--step_size',type=int,default=100,help="steplr's step size")
    parser.add_argument('--gamma',type=float,default=0.1,help="learning rate decay")
    # recorder and logging
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='../checkpoints/BNN', type=str)
    parser.add_argument('--postfix',
                        help='model folder postfix',
                        default='1', type=str)
    parser.add_argument('--save_freq',type=int,default=1,help="how many epoch to save model")
    parser.add_argument('--print_freq',type=int,default=150,help="print_freq")
    # gpus
    parser.add_argument('--gpus',type=int,default=1,help="gpu number")
    parser.add_argument('--manualSeed',type=int,default=0,help="default init seed")

    args = parser.parse_args()
    return args




def main():
    # args
    args=getArgs()
    if args.arch != "resnet18":
        args.steplist = [150,220,260]
    else:
        args.steplist = [40,80,120]
    # logging
    projectName="{}_{}_{}_{}_{}_{}".format(args.model.lower(),args.datasets,
                                    args.epochs,args.batch_size,
                                    args.lr,args.postfix)
    modelDir=os.path.join(args.save_dir,projectName)
    logger = get_logger(modelDir)
    with open(os.path.join(modelDir,"args.yaml"), "w") as yaml_file:  # dump experiment config
        yaml.dump(args, yaml_file)

    pymonitor = ProgressMonitor(logger)
    tbmonitor = TensorBoardMonitor(logger, modelDir)
    monitors = [pymonitor, tbmonitor]

    # dataloader
    trainLoader = getDataloader(args.datasets,"train",args.batch_size,
                    args.workers,args.pin_memory,args.cutout)
    valLoader = getDataloader(args.datasets,"val",args.batch_size,
                        args.workers,args.pin_memory,args.cutout)

    # device init 
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    args.device = torch.device('cpu')
    # model 
    model = ARCH_DICT[args.arch].__dict__[args.model](pretrained=args.pretrained, 
                                        progress=True,
                                        num_classes=args.class_num)
    logger.info("model is:{} \n".format(model))


    # loss and optimazer
    criterion = nn.CrossEntropyLoss()

    # best recoder
    perf_scoreboard = PerformanceScoreboard(args.num_best_scores)

    # # resume
    # assert args.resume == True
    model,extras,start_epoch=loadCheckpoint(args.resume_path,model,args)
    _,_,_=extras["optimizer"],extras['scheduler'],extras["perf_scoreboard"]

    # # just eval model
    tp1,tp5,loss = validate(valLoader, model, criterion, args)
    print("origianl model eval :{} ".format(tp1,tp5,loss))


    # 量化bits[8,6,5,4,3,2,1]
    # 逐层进行量化
    # vggsmall,resnet18,resnet20
    careops = ["Conv2d"]
    testbits = [8,7,6,5,4,3,2,1]
    quantityOps = findQuantityOps(model,careops)
    # 生成各个bit的逐层量化方案
    quantityinfo = createQuantityInfo(quantityOps,testbits)
    # 根据生成的量化方案执行量化分析，查看逐层量化时的准确率和loss变化情况
    quantityAnalysis(model,quantityOps,quantityinfo, valLoader, criterion, args)



def createQuantityInfo(quantityOps,testbits):
    '''
        quantityOps: care ops [full_name....]
        testbits: [bit1,bit2 .....]
    '''
    bitTestInfo= OrderedDict()
    for bit in testbits:
        print("Full quantity weight and activation: {} bit".format(bit))
        keyOpsInfo = OrderedDict()
        for index in range(len(quantityOps)):
            keyOpsInfo[index] = {}
            wbits = [32 for i in range(len(quantityOps)) ]
            wbits[index] = bit
            abits = [32 for i in range(len(quantityOps))]
            abits[index] = bit
            keyOpsInfo[index]["w"] = wbits
            keyOpsInfo[index]["a"] = abits
        bitTestInfo[bit] = keyOpsInfo
    return bitTestInfo


def findQuantityOps(model,careops):
    '''
    input model
    careops: ["Conv2d"]
    '''
    quantityOps=[]
    for name,module in model.named_modules():
        if type(module).__name__ in careops:
            quantityOps.append(name)
    return quantityOps


# 其实可以在这里直接进行模型op的替换
def getQuantityOps(model,quantityOps,wbits,abits):
    assert len(quantityOps) == len(wbits) and len(wbits) == len(abits)
    modulereplaces = {}
    for name,module in model.named_modules():
        if name in quantityOps:
            index =  quantityOps.index(name)
            assert type(module) in QuanModuleMapping.keys()
            newops = QuanModuleMapping[type(module)](
                    module,
                    wbit = wbits[index],
                    abit = abits[index]
                )
            modulereplaces[name] = newops
    return modulereplaces

def replaceModuleByNames(model,quantityOps,modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m and full_name in quantityOps:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model

def quantityAnalysis(model,quantityOps,quantityinfo,valLoader, criterion, args):
    '''
        original model
        quantity ops
        quantity info 
    '''

    bits = list(quantityinfo.keys())
    layers = list(quantityinfo[bits[0]].keys())
    validateDict = {}
    for qbit in bits:
        for lindex in layers:
            # print("Quantity:{} bit and Layer:{} ".format(qbit,lindex))
            wadict = quantityinfo[qbit][lindex]
            wbits,abits = wadict["w"],wadict["a"]
            modulereplaces= getQuantityOps(model,quantityOps,wbits,abits)
            newmodel = replaceModuleByNames(model.copy(),quantityOps,modulereplaces)
            tp1,tp5,loss = validate(valLoader, newmodel, criterion, args)
            print("Bit:{}  layer:{}  Top1:{} ".format(qbit,lindex,tp1))




def validate(data_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
    return top1.avg, top5.avg, losses.avg




if __name__ == '__main__':
    main()