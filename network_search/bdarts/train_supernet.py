import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search_v1 import Network


import utils
sys.path.append("../../")
from  utils import *


def get_args():
  parser = argparse.ArgumentParser("Binary nerual networks search")
  parser.add_argument('--dataset', type=str, default='../data', help='location of the data corpus')
  parser.add_argument('--class_num', type=int, default=10, help='num of dataset class')
  parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--workers', type=int, default=4, help='workers number')
  parser.add_argument('--arch_after', type=int, default=200, help='arch_after of training epochs')

  parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
  parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

  parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
  parser.add_argument('--layers', type=int, default=8, help='total number of layers')
  parser.add_argument('--nodes', type=int, default=4, help='middle nodes')
  parser.add_argument('--group', type=int, default=12, help='group numbers')
  parser.add_argument('--stem_multiplier', type=int, default=3, help='stem_multiplier')
  parser.add_argument('--init_type',default="kaiming",help="weight init func")

  parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
  parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
  parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')

  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

  parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
  parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

  parser.add_argument('--num_best_scores',type=int,default=5,help="num_best_scores")
  parser.add_argument('--save_freq',type=int,default=1,help="how many epoch to save model")
  parser.add_argument('--print_freq', type=float, default=150, help='report frequency')
  parser.add_argument('--gpus',type=int,default=1,help="gpu number")

  parser.add_argument('--save', type=str, default='EXP', help='experiment name')
  parser.add_argument('--resume', action='store_true', default=False, help='use resume')
  parser.add_argument('--resume_path', type=str, default='saved_models', help='path to save the model')
  
  args = parser.parse_args()

  return args


def main():
  args = get_args()

  # get log 
  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
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
  if args.seed is None:
      args.seed = random.randint(1, 10000)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
  args.multi_gpu = args.gpus > 1 and torch.cuda.is_available()
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
  model = Network(C=args.init_channels,num_classes=args.class_num,layers=args.layers,steps=args.nodes,multiplier=args.nodes,
    stem_multiplier=args.stem_multiplier,group=args.group)
  if args.multi_gpu:
      logger.info('use: %d gpus', args.gpus)
      model = nn.DataParallel(model)
  model = model.freeze_arch_parameters()
  model = model.to(args.device)
  criterion = criterion.to(args.device)
  logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # get dataloader
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  traindata = dset.CIFAR10(root=args.dataset, train=True, download=False, transform=train_transform)
  valdata = dset.CIFAR10(root=args.dataset, train=False, download=False, transform=valid_transform)
  trainLoader = torch.utils.data.DataLoader(
      traindata, batch_size=args.batch_size,
      pin_memory=True,shuffle=True,num_workers=args.workers)
  valLoader = torch.utils.data.DataLoader(
      valdata, batch_size=args.batch_size,
      pin_memory=True, num_workers=args.workers)

  # weight optimizer and struct parameters /mask optimizer
  optimizer = torch.optim.SGD(
      model.weight_parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  perf_scoreboard = PerformanceScoreboard(args.num_best_scores)
  # resume 
  start_epoch=0
  if args.resume:
      if os.path.isfile(args.resume_path):
          model,extras,start_epoch=loadCheckpoint(args.resume_path,model,args)
          optimizer,scheduler,perf_scoreboard=extras["optimizer"],extras['scheduler'],extras["perf_scoreboard"]
      else:
          raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

  for epoch in range(start_epoch,args.epochs):
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    t_top1, t_top5, t_loss = train(trainLoader, valLoader, model, criterion, epoch, optimizer, monitors, args,logger)
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
        saveCheckpoint(epoch, "search", model,
                        {
                        'scheduler': scheduler,
                         'optimizer': optimizer,
                         'perf_scoreboard' : perf_scoreboard
                         }, 
                        is_best,os.path.join(args.save,"ckpts"))
    # update lr
    scheduler.step()



def train(train_queue, valid_queue, model,criterion,epoch,optimizer,monitors, args,logger):
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
      inputs =  Variable(inputs, requires_grad=False).to(args.device)
      targets = Variable(targets, requires_grad=False).to(args.device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      optimizer.zero_grad()
      loss.backward()
      if args.grad_clip:
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
      optimizer.step()

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

