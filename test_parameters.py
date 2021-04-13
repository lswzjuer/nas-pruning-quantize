# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2021-01-03 22:05:10
# @Last Modified by:   liusongwei
# @Last Modified time: 2021-01-03 22:13:30

# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-11-28 11:14:04
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-12-10 02:11:42
import os
import sys
import time
import glob
import numpy as np
import torch
import tools
import logging
import argparse
import torch.nn as nn
import binary_genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from binary_model import NetworkCIFAR as Network

import matplotlib.pyplot as plt
from collections import Counter,OrderedDict
from utils import *
from datasets.classification import getDataloader
import math
import yaml

# Count the calculations and parameters of common models
def paramsFlopsCounter(model,num_classes=10,input_shape=(3,32,32)):
    logger=get_logger("./")
    pa1=getParams(model)
    fl1=getFlops(model,input_shape)
    fl2,pa2=get_model_complexity_info(model,input_shape,True)
    #logger.info("{}  v1: {}--{} ".format(model,pa1,fl1))
    logger.info("{}  v1: {}--{}  v2: {}--{}".format(modelname,pa1,fl1,pa2,fl2))



def get_args():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=150, help='report frequency')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--group', type=int, default=12, help='group numbers')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    
    parser.add_argument('--save', type=str, default='/checkpoints', help='experiment name')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='waadv_stage3_final', help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    parser.add_argument('--tmp_data_dir', type=str, default='/share/jjchu/source_code/DataSets', help='temp data dir')
    
    parser.add_argument('--note', type=str, default='try', help='note for this run')
    parser.add_argument('--cifar100', action='store_true', default=False, help='if use cifar100')
    parser.add_argument('--eval', action='store_true', default=False, help='eval')
    parser.add_argument('--resume', action='store_true', default=False, help='resume')
    parser.add_argument('--resume_path',type=str,default="./checkpoint/model_xx",help="traing resume_path")  
    parser.add_argument('--gpus', type=int, default=1, help='gpus numbers')

    parser.add_argument('--num_best_scores',type=int,default=5,help="num_best_scores")
    parser.add_argument('--optimizer',type=str,default="adam",choices=["adam","sgd","radam"],help="optimizer")
    parser.add_argument('--scheduler',type=str,default="cos",choices=["warm_up_cos","cos","step","mstep"],help="scheduler")
    parser.add_argument('--step_size',type=int,default=100,help="steplr's step size")
    parser.add_argument('--gamma',type=float,default=0.1,help="learning rate decay")

    args = parser.parse_args()

    return args




def main():
    # get log
    args = get_args()
    # load model and loss func
    logger=get_logger("./")
    args.device = torch.device("cpu")
    genotype = eval("binary_genotypes.%s" % args.arch)
    logger.info('---------Genotype---------')
    logger.info(genotype)
    logger.info('--------------------------')
    model = Network(args.init_channels, 1000, args.layers, args.auxiliary, genotype,args.group)
    model = model.to(args.device)
    logging.info("param size = %fMB", tools.count_parameters_in_MB(model))
    paramsFlopsCounter(model,num_classes=1000,input_shape=(3,224,224))


if __name__ == '__main__':
    main()