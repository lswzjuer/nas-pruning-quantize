# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-11-28 11:14:04
# @Last Modified by:   liusongwei
# @Last Modified time: 2021-01-03 22:09:20
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


import yaml
import math
sys.path.append("../../")
from utils import *


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
    
    parser.add_argument('--save', type=str, default='/share/jjchu/source_code/pbdarts/checkpoints', help='experiment name')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='BATS', help='which architecture to use')
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
    args.save = '{}/eval-{}-{}'.format(args.save,args.note,time.strftime("%Y%m%d-%H%M%S"))
    # if not os.path.exists(args.save):
    #     os.path.mkdir(args.save)
    tools.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('Train Search')
    logger.addHandler(fh)

    # monitor
    pymonitor = ProgressMonitor(logger)
    tbmonitor = TensorBoardMonitor(logger, args.save)
    monitors = [pymonitor, tbmonitor]

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
    args.device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
    setting = {k: v for k, v in args._get_kwargs()}
    logger.info(setting)
    with open(os.path.join(args.save,"args.yaml"), "w") as yaml_file:  # dump experiment config
        yaml.dump(args, yaml_file)

    pymonitor = ProgressMonitor(logger)
    tbmonitor = TensorBoardMonitor(logger, args.save)
    monitors = [pymonitor, tbmonitor]

    if args.cifar100:
        CIFAR_CLASSES = 100
        data_folder = 'cifar-100-python'
    else:
        CIFAR_CLASSES = 10
        data_folder = 'cifar-10-batches-py'

    # load model and loss func 
    genotype = eval("binary_genotypes.%s" % args.arch)
    logger.info('---------Genotype---------')
    logger.info(genotype)
    logger.info('--------------------------')
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype,args.group)
    model = model.to(args.device)
    logging.info("param size = %fMB", tools.count_parameters_in_MB(model))

    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)


    if args.cifar100:
        train_transform, valid_transform = tools._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = tools._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)

    trainLoader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valLoader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    


    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                    betas=(0.9, 0.999), 
                                    weight_decay=args.weight_decay)

    elif args.optimizer.lower() == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.learning_rate,
                                    betas=(0.9, 0.999), 
                                    weight_decay=args.weight_decay)
    else:
        NotImplementedError()


    if args.scheduler.lower() == 'warm_up_cos':
        warm_up_epochs = 5
        warm_up_with_adam = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
         else 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_adam)

    elif args.scheduler.lower() == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,eta_min=0,last_epoch=-1)

    elif args.scheduler.lower() == "step":
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=args.step_size,gamma=0.1,last_epoch=-1)
    
    elif  args.scheduler.lower() == "mstep":
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steplist, gamma=args.gamma)
    else:
        NotImplementedError()


    # best recoder
    perf_scoreboard = PerformanceScoreboard(args.num_best_scores)

    # resume 
    start_epoch=0
    if args.resume:
        if os.path.isfile(args.resume_path):
            model,extras,start_epoch=loadCheckpoint(args.resume_path,model,args)
            optimizer,perf_scoreboard=extras["optimizer"],extras["perf_scoreboard"]
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

    # just eval model
    if args.eval:
        validate(valLoader, model, criterion, -1, monitors, args,logger)
    else:
        # resume training or pretrained model, we should eval model firstly.
        if args.resume:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = validate(valLoader, model, criterion,
                                             start_epoch - 1, monitors, args,logger)
            l,board=perf_scoreboard.update(top1, top5, start_epoch - 1)
            for idx in range(l):
                score = board[idx]
                logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                                idx + 1, score['epoch'], score['top1'], score['top5'])

        # start training
        for _ in range(start_epoch):
            scheduler.step()

        for epoch in range(start_epoch, args.epochs):
            drop_prob = args.drop_path_prob * epoch / args.epochs
            model.drop_path_prob = drop_prob
            logger.info('>>>> Epoch {} Lr {} Drop:{} '.format(epoch,
                                                            optimizer.param_groups[0]['lr'],
                                                            drop_prob))

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
            if (epoch+1)% 5==0:
                saveCheckpoint(epoch, "search", model,
                                {
                                # 'scheduler': scheduler,
                                 'optimizer': optimizer,
                                 'perf_scoreboard' : perf_scoreboard
                                 }, 
                                is_best,os.path.join(args.save,"ckpts"))
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

        outputs,outputs_aux = model(inputs)
        loss = criterion(outputs, targets)
        if args.auxiliary:
            loss_aux = criterion(outputs_aux, targets)
            loss += args.auxiliary_weight*loss_aux

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.report_freq == 0:
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

            outputs,_ = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.report_freq == 0:
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
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    print('Total train time: %ds', duration)
