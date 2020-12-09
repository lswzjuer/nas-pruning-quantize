# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-19 20:57:00
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-11 16:53:46


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
import mobilenetv2_dr as networka
import resnet18_dr as networkb
import resnet20_dr as networkc
import vggsmall_dr as networkd
from attack import AttackPGD



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
                        default='../checkpoints/DOREAFA', type=str)
    parser.add_argument('--postfix',
                        help='model folder postfix',
                        default='1', type=str)
    parser.add_argument('--save_freq',type=int,default=1,help="how many epoch to save model")
    parser.add_argument('--print_freq',type=int,default=150,help="print_freq")
    # gpus
    parser.add_argument('--gpus',type=int,default=1,help="gpu number")
    parser.add_argument('--manualSeed',type=int,default=0,help="default init seed")

    # weight and activate Quantization bits wide
    parser.add_argument('--wbit',type=int,default=8,help="gpu number")
    parser.add_argument('--abit',type=int,default=8,help="gpu number")

    args = parser.parse_args()
    return args



def main():
    # args
    args=getArgs()
    if args.arch in["resnet18","mobilenetv2","mobilenetv1","ghostnet"]:
        args.steplist = [30,60,90]
        # args.steplist = [40,80,120]
    else:
        args.steplist = [150,220,260]
    # logging
    projectName="{}_{}_{}_{}_{}_{}and{}_{}".format(args.model.lower(),args.datasets,
                                    args.epochs,args.batch_size,
                                    args.lr,args.wbit,args.abit,args.postfix)
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
    args.use_cuda= args.gpus>0 and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.benchmark = True

    # model 
    model = ARCH_DICT[args.arch].__dict__[args.model](pretrained=args.pretrained, 
                                        progress=True,
                                        wbit=args.wbit,
                                        abit=args.abit,
                                        num_classes=args.class_num)
    logger.info("model is:{} \n".format(model))

    # loss and optimazer
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        logger.info("load model and criterion to gpu !")
        model=model.to(args.device)
        criterion=criterion.to(args.device)

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.999), 
                                    weight_decay=args.weight_decay)
    else:
        NotImplementedError()

    if args.scheduler.lower() == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,eta_min=0,last_epoch=-1)

    elif args.scheduler.lower() == "step":
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=args.step_size,gamma=0.1,last_epoch=-1)
    
    elif args.scheduler.lower() == "mstep":
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steplist, gamma=args.gamma)
    
    else:
        NotImplementedError()


    # resume 
    start_epoch=0
    if args.resume:
        if os.path.isfile(args.resume_path):
            model,extras,start_epoch=loadCheckpoint(args.resume_path,model,args)
            _,_,_=extras["optimizer"],extras['scheduler'],extras["perf_scoreboard"]
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

    model_adv = AttackPGD(model,None)

    # just eval model
    if args.eval:
        v1, v5, loss = validate(valLoader, model_adv, criterion, args, adv = False)
        logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', v1, v5, loss)

        v1, v5, loss = validate(valLoader, model_adv, criterion, args, adv = True)
        logger.info('==> PGD20 Top1: %.3f    Top5: %.3f    Loss: %.3f\n', v1, v5, loss)   

    else:
        # adv training  some epochs and eval 
        for epoch in range(args.epochs):
            
            logger.info('>>>> Epoch {} Lr {} '.format(epoch,optimizer.param_groups[0]['lr'] ))

            t_top1, t_top5, t_loss = train(trainLoader, model_adv, criterion, optimizer,
                                                   scheduler,args,adv = True)
            logger.info('ADV Training ==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                        t_top1, t_top5, t_loss)

            v_top1, v_top5, v_loss = validate(valLoader, model_adv, criterion,args,adv = True)

            logger.info('ADV Validation ==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                        v_top1, v_top5, v_loss)









    #     best_acc = 0
    #     for epoch in range(start_epoch, args.epochs):
            
    #         logger.info('>>>> Epoch {} Lr {} '.format(epoch,optimizer.param_groups[0]['lr'] ))

    #         t_top1, t_top5, t_loss = train(trainLoader, model_adv, criterion, optimizer,
    #                                                scheduler,args,adv = True)
    #         logger.info('ADV Training ==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
    #                     t_top1, t_top5, t_loss)

    #         v_top1, v_top5, v_loss = validate(valLoader, model_adv, criterion,args,adv = True)

    #         logger.info('ADV Validation ==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
    #                     v_top1, v_top5, v_loss)

    #         tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
    #         tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
    #         tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)


    #         is_best = True if v_top1 > best_acc else False
    #         best_acc = v_top1 if v_top1 > best_acc else best_acc

    #         # save model
    #         if epoch% args.save_freq==0:
    #             saveCheckpoint(epoch, args.model, model_adv.model,
    #                             {
    #                             'scheduler': scheduler,
    #                              'optimizer': optimizer,
    #                              }, 
    #                             is_best,os.path.join(modelDir,"ckpts"))
    #         # update lr
    #         scheduler.step()

    #     logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
    #     v_top1, v_top5, v_loss = validate(valLoader, model_adv, criterion,args,adv = True)
    #     logger.info('ADV Validation ==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
    #                 v_top1, v_top5, v_loss)

    # tbmonitor.writer.close()  # close the TensorBoard
    # logger.info('Program completed successfully ... exiting ...')
    # logger.info('If you have any questions or suggestions, please visit: github.com/lswzjuer/nas-pruning-quantize')



def train(train_loader, model, criterion, optimizer, scheduler, args, adv = True):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        if not adv:
            model.attack = False
            outputs, _ = model(inputs,targets)
            loss = criterion(outputs, targets)
        else:
            model.attack = False
            outputs1, _ = model(inputs,targets)
            model.attack = True
            outputs, _ = model(inputs,targets)
            loss = (criterion(outputs1, targets) + criterion(outputs, targets) )/2.0
            
        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, criterion, args,adv=False):
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
            if not adv:
                model.attack = False
                outputs, _ = model(inputs,targets)
                loss = criterion(outputs, targets)
            else:
                model.attack = True
                outputs, _ = model(inputs,targets)
                loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
    
    return top1.avg, top5.avg, losses.avg




if __name__ == '__main__':
    main()