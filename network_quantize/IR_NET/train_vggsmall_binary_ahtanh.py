# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-21 13:02:40
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-21 15:58:56

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
import vggsmall_ahtanh as vgg

sys.path.append("../../")
from  utils import *
from datasets.classification import getDataloader


Tdict={
    "vgg_small_1w1a_ahtanhlayer" : [1e-1, 1e1],
    "vgg_small_1w1a_ahtanhlayershared" : [1e-1, 1e1],
    "vgg_small_1w1a_ahtanhchannel" : [1e-1, 1e1],
    "vgg_small_1w32a_ahtanh" : [1e-1, 1e1],
}


def getArgs():
    parser=argparse.ArgumentParser("Train network in cifar10/100/svhn/mnist/tiny_imagenet")
    # model and train setting
    parser.add_argument('--model',default="vgg_small_1w1a_ahtanhlayershared",
                                choices=['vgg_small_1w1a_ahtanhlayer','vgg_small_1w1a_ahtanhlayershared',
                                        'vgg_small_1w1a_ahtanhchannel','vgg_small_1w32a_ahtanh'],
                                help="binary resnet models")
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
    parser.add_argument('--lr', default=0.007, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--workers',type=int,default=4,help="dataloader num_workers")
    parser.add_argument("--pin_memory",type=bool,default=True,help="dataloader cache ")
    parser.add_argument('--cutout',default=False, action='store_true')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume',action='store_true',default=False,help="traing resume")
    parser.add_argument('--resume_path',type=str,default="./checkpoint/model_xx",help="traing resume_path")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--num_best_scores',type=int,default=10,help="num_best_scores")
    parser.add_argument('--optimizer',type=str,default="sgd",choices=["adam","sgd"],help="optimizer")
    parser.add_argument('--scheduler',type=str,default="cos",choices=["cos","steplr"],help="scheduler")
    parser.add_argument('--step_size',type=int,default=100,help="steplr's step size")

    # recorder and logging
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='../checkpoints', type=str)
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
    args.use_cuda= args.gpus>0 and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.benchmark = True

    # model 
    model = vgg.__dict__[args.model](pretrained=args.pretrained, 
                                        progress=True,
                                        num_classes=args.class_num)

    logger.info("model is:{} \n".format(model))

    if torch.cuda.device_count() > 1 and args.use_cuda:
        logger.info('use: %d gpus', torch.cuda.device_count())
        model = nn.DataParallel(model)

    # loss and optimazer
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        logger.info("load model and criterion to gpu !")
        model=model.to(args.device)
        criterion=criterion.to(args.device)


    # all_parameters = model.parameters()
    # act_params=[]
    # for pname,p in model.named_parameters():
    #     if "nonlinear" in pname:
    #         act_params.append(p)
    # act_params_id= list(map(id,act_params))
    # other_parameters= list(filter(lambda p: id(p) not in act_params_id, all_parameters))

    logger.info("Act layer params :\n {}".format(model.alpha_parameters()))

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
                    [
                    {"params": model.other_parameters(),"lr":args.lr,"weight_decay":args.weight_decay},
                    {"params": model.alpha_parameters(), "lr": 0.001, "weight_decay": 0.0}
                    ],
                    momentum=args.momentum )

    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
                    [
                    {"params": model.other_parameters(),"lr":args.lr,"weight_decay":args.weight_decay},
                    {"params": model.alpha_parameters(), "lr": 0.001, "weight_decay": 0.0}
                    ],
                    betas=(0.9, 0.999))
    else:
        NotImplementedError()

    if args.scheduler.lower() == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,eta_min=0,last_epoch=-1)

    elif args.scheduler.lower() == "steplr":
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=args.step_size,gamma=0.1,last_epoch=-1)
    else:
        NotImplementedError()


    # best recoder
    perf_scoreboard = PerformanceScoreboard(args.num_best_scores)

    # resume 
    start_epoch=0
    if args.resume:
        if os.path.isfile(args.resume_path):
            model,extras,start_epoch=loadCheckpoint(args.resume_path,model,args)
            optimizer,scheduler,perf_scoreboard=extras["optimizer"],extras['scheduler'],extras["perf_scoreboard"]
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

    T_min, T_max = Tdict[args.model]
    logging.info("Tmin:{} Tmax:{}".format(T_min,T_max))

    # update k,t
    def Log_UP(K_min, K_max, epoch):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float()

    # just eval model
    if args.eval:
        validate(valLoader, model, criterion, -1, monitors, args,logger)
    else:
        # resume training or pretrained model, we should eval model firstly.
        if args.resume or args.pretrained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = validate(valLoader, model, criterion,
                                             start_epoch - 1, monitors, args,logger)
            l,board=perf_scoreboard.update(top1, top5, start_epoch - 1)
            for idx in range(l):
                score = board[idx]
                logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                                idx + 1, score['epoch'], score['top1'], score['top5'])

        # start training 
        for epoch in range(start_epoch, args.epochs):

            t = Log_UP(T_min, T_max, epoch)
            if (t < 1):
                k = 1 / t
            else:
                k = torch.tensor([1]).float()
            if args.use_cuda:
                k=k.cuda()
                t=t.cuda()

            model.conv0.k = k
            model.conv1.k = k
            model.conv2.k = k
            model.conv3.k = k
            model.conv4.k = k
            model.conv5.k = k
            model.conv0.t = t
            model.conv1.t = t
            model.conv2.t = t
            model.conv3.t = t
            model.conv4.t = t
            model.conv5.t = t


            logger.info('>>>> Epoch {} Lr {} K:{}  T:{}'.format(epoch,
                                                            optimizer.param_groups[0]['lr'],
                                                            k,t))

            t_top1, t_top5, t_loss = train(trainLoader, model, criterion, optimizer,
                                                   scheduler, epoch, monitors, args,logger)
            v_top1, v_top5, v_loss = validate(valLoader, model, criterion, epoch, monitors, args,logger)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            l,board=perf_scoreboard.update(v_top1, v_top5, epoch)
            for idx in range(l):
                score = board[idx]
                logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                                idx + 1, score['epoch'], score['top1'], score['top5'])

            is_best = perf_scoreboard.is_best(epoch)
            # save model
            if epoch% args.save_freq==0:
                saveCheckpoint(epoch, args.model, model,
                                {
                                'scheduler': scheduler,
                                 'optimizer': optimizer,
                                 'perf_scoreboard' : perf_scoreboard
                                 }, 
                                is_best,os.path.join(modelDir,"ckpts"))
            # update lr
            scheduler.step()

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        validate(valLoader, model, criterion, -1, monitors, args,logger)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/lswzjuer/nas-pruning-quantize')



def train(train_loader, model, criterion, optimizer, scheduler, epoch, monitors, args,logger):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))


        optimizer.zero_grad()
        loss.backward()

        # # vis alpha param`s value and it`s grad
        # for pname,p in model.named_parameters():
        #     if "nonlinear" in pname:
        #         print("{}  oldvalue:{}  currentgrad:{}".format(pname, p, p.grad))

                
        # update alpha and weights
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr']
                })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, criterion, epoch, monitors, args,logger):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    end_time = time.time()
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
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.print_freq == 0:
                for m in monitors:
                    m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg




if __name__ == '__main__':
    main()