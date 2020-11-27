import os
import sys
import yaml
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from binary_model_search import Network


import tools
import math
sys.path.append("../../")
from utils import *



def get_args():
  parser = argparse.ArgumentParser("Binary nerual networks search")
  parser.add_argument('--dataset_name', type=str, default='cifar10', help='dataset name')
  parser.add_argument('--dataset', type=str, default=r'F:\source_code\DataSets', help='location of the data corpus')
  parser.add_argument('--class_num', type=int, default=10, help='num of dataset class')
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--workers', type=int, default=4, help='workers number')
  parser.add_argument('--arch_after', type=int, default=5, help='arch_after of training epochs')

  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

  parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
  parser.add_argument('--layers', type=int, default=8, help='total number of layers')
  parser.add_argument('--nodes', type=int, default=4, help='middle nodes')
  parser.add_argument('--group', type=int, default=12, help='group numbers')
  parser.add_argument('--stem_multiplier', type=int, default=3, help='stem_multiplier')
  parser.add_argument('--init_type',default="kaiming",help="weight init func")

  parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
  parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
  parser.add_argument('--drop_rate', type=float, default=0.3, help='drop path probability')

  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--gpus', type=int, default=1, help='gpus number')
  parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

  parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
  parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

  parser.add_argument('--num_best_scores',type=int,default=5,help="num_best_scores")
  parser.add_argument('--save_freq',type=int,default=1,help="how many epoch to save model")
  parser.add_argument('--print_freq', type=float, default=150, help='report frequency')


  parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
  parser.add_argument('--save', type=str, default='EXP', help='experiment name')

  parser.add_argument('--resume',action='store_true',default=False,help="traing resume")
  parser.add_argument('--resume_path',type=str,default="./checkpoint/model_xx",help="traing resume_path")


  args = parser.parse_args()

  return args


def main():

  args = get_args()
  # get log 
  args.save = '{}/search-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

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

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(args.device)
  model = Network(C=args.init_channels,num_classes=args.class_num,layers=args.layers,criterion=criterion,
    steps=args.nodes,multiplier=args.nodes,stem_multiplier=args.stem_multiplier,group=args.group)
  model = model.to(args.device)
  logger.info("param size = %fMB", tools.count_parameters_in_MB(model))

  # get dataloader
  if args.dataset_name == "cifar10":
    train_transform, valid_transform = tools._data_transforms_cifar10(args)
    traindata = dset.CIFAR10(root=args.dataset, train=True, download=False, transform=train_transform)
    valdata = dset.CIFAR10(root=args.dataset, train=False, download=False, transform=valid_transform)
  
  else:
    train_transform, valid_transform = tools._data_transforms_mnist(args)
    traindata = dset.MNIST(root=args.dataset, train=True, download=False, transform=train_transform)
    valdata = dset.MNIST(root=args.dataset, train=False, download=False, transform=valid_transform)

  num_train = len(traindata)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))
  train_queue = torch.utils.data.DataLoader(
      traindata, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=args.workers)
  valid_queue = torch.utils.data.DataLoader(
      traindata, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=args.workers)
  valLoader = torch.utils.data.DataLoader(
      valdata, batch_size=args.batch_size,
      pin_memory=True, num_workers=args.workers)

  # weight optimizer and struct parameters /mask optimizer
  optimizer_w = torch.optim.SGD(
      model.weight_parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_w, float(args.epochs), eta_min=args.learning_rate_min)

  optimizer_a = torch.optim.Adam(model.arch_parameters(),
          lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)


  perf_scoreboard = PerformanceScoreboard(args.num_best_scores)

  # resume 
  start_epoch=0
  if args.resume:
      if os.path.isfile(args.resume_path):
          model,extras,start_epoch=loadCheckpoint(args.resume_path,model,args)
          optimizer_w,optimizer_a,perf_scoreboard=extras["optimizer_w"],extras["optimizer_a"],extras["perf_scoreboard"]
      else:
          raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

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
      scheduler_w.step()

  for epoch in range(args.epochs):
    lr = scheduler_w.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)


    if epoch < args.arch_after:
        model.p = float(args.drop_rate) * (args.epochs - epoch - 1) / args.epochs
        model.update_p()
        t_top1, t_top5, t_loss = train(train_queue, valid_queue, model, criterion, epoch, optimizer_w, optimizer_a, 
                                      args,monitors,logger,train_arch=False)
    else:
        model.p = float(args.drop_rate) * np.exp(-(epoch - args.arch_after) * 0.2) 
        model.update_p()
        t_top1, t_top5, t_loss = train(train_queue, valid_queue, model, criterion, epoch, optimizer_w, optimizer_a, 
                                      args,monitors,logger,train_arch=True)  
    
    v_top1, v_top5, v_loss = validate(valLoader, model, criterion, epoch,args,monitors,logger)

    l,board=perf_scoreboard.update(v_top1, v_top5, epoch)
    logger.info("normal: \n{}".format(model.alphas_normal.data.cpu().numpy()))
    logger.info("reduce: \n{}".format(model.alphas_reduce.data.cpu().numpy()))
    logger.info('Genotypev: {}'.format(model.genotype()))
    is_best = perf_scoreboard.is_best(epoch)
    # save model
    if epoch% args.save_freq==0:
        saveCheckpoint(epoch, args.model, model,
                        {
                         'optimizer_w': optimizer_w,
                         'optimizer_a': optimizer_a,
                         'perf_scoreboard' : perf_scoreboard
                         }, 
                        is_best,os.path.join(args.save,"ckpts"))
    # update lr
    scheduler_w.step()



def train(train_queue, valid_queue, model,criterion,epoch,optimizer_w, optimizer_a,
    args,monitors,logger,train_arch=True):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_queue.sampler)
    batch_size = train_queue.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_queue):
      # weight train samples
      inputs =  inputs.to(args.device)
      targets = targets.to(args.device)

      # get a random minibatch from the search queue with replacement
      if train_arch:
        try:
          input_search, target_search = next(valid_queue)
        except:
          input_search, target_search = next(iter(valid_queue))
        input_search = input_search.to(args.device)
        target_search = target_search.to(args.device)
        # updata structure parameters or masks
        optimizer_a.zero_grad()
        output_search = model(input_search)
        arch_loss = criterion(output_search, target_search)
        arch_loss.backward()
        optimizer_a.step()

      outputs = model(inputs)
      loss = criterion(outputs, targets)
      optimizer_w.zero_grad()
      loss.backward()
      if args.grad_clip:
        nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)
      optimizer_w.step()
      acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
      losses.update(loss.item(), inputs.size(0))
      top1.update(acc1.item(), inputs.size(0))
      top5.update(acc5.item(), inputs.size(0))
      batch_time.update(time.time() - end_time)
      end_time = time.time()

      if (batch_idx + 1) % args.print_freq == 0:
          for m in monitors:
              m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                  'Loss': losses,
                  'Top1': top1,
                  'Top5': top5,
                  'BatchTime': batch_time,
                  'LR': optimizer_w.param_groups[0]['lr']
              })
    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg



def validate(data_loader, model, criterion, epoch,args,monitors,logger):
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

