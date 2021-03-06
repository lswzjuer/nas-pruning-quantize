import os
import sys
import yaml
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
from torch.optim.lr_scheduler import StepLR
from fista import FISTA
from model import Discriminator

import tools
sys.path.append("../../")
from  utils import *


def get_args():
  parser = argparse.ArgumentParser("Binary nerual networks search")
  parser.add_argument('--dataset_name', type=str, default='cifar10', help='dataset name')
  parser.add_argument('--dataset', type=str, default='../data', help='location of the data corpus')
  parser.add_argument('--class_num', type=int, default=10, help='num of dataset class')
  parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--workers', type=int, default=4, help='workers number')
  parser.add_argument('--arch_after', type=int, default=10, help='arch_after of training epochs')
  parser.add_argument('--lr_decay_step', type=int, default=60, help='lr_decay_step')

  parser.add_argument('--learning_rate', type=float, default=1e-2, help='init learning rate')
  parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

  parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
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
  parser.add_argument('--arch_learning_rate', type=float, default=1e-2, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')
  parser.add_argument('--sparse_lambda', type=float, default=0.6, help='sparse_lambda')

  parser.add_argument('--num_best_scores',type=int,default=5,help="num_best_scores")
  parser.add_argument('--save_freq',type=int,default=1,help="how many epoch to save model")
  parser.add_argument('--print_freq', type=float, default=150, help='report frequency')
  parser.add_argument('--gpus',type=int,default=1,help="gpu number")

  parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
  parser.add_argument('--save', type=str, default='EXP', help='experiment name')
  parser.add_argument('--resume', action='store_true', default=False, help='use resume')
  parser.add_argument('--resume_path', type=str, default='saved_models', help='path to save the model')
  
  args = parser.parse_args()

  return args


def main():
  args = get_args()

  # get log 
  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  # get dataloader
  if args.dataset_name == "cifar10":
    train_transform, valid_transform = tools._data_transforms_cifar10(args)
    traindata = dset.CIFAR10(root=args.dataset, train=True, download=False, transform=train_transform)
    valdata = dset.CIFAR10(root=args.dataset, train=False, download=False, transform=valid_transform)
  else:
    train_transform, valid_transform = tools._data_transforms_mnist(args)
    traindata = dset.MNIST(root=args.dataset, train=True, download=False, transform=train_transform)
    valdata = dset.MNIST(root=args.dataset, train=False, download=False, transform=valid_transform)
  trainLoader = torch.utils.data.DataLoader(
      traindata, batch_size=args.batch_size,
      pin_memory=True,shuffle=True,num_workers=args.workers)
  valLoader = torch.utils.data.DataLoader(
      valdata, batch_size=args.batch_size,
      pin_memory=True, num_workers=args.workers)

  # load pretrained model
  model_t = Network(C=args.init_channels,num_classes=args.class_num,layers=args.layers,steps=args.nodes,multiplier=args.nodes,
    stem_multiplier=args.stem_multiplier,group=args.group)
  model_t,_,_ = loadCheckpoint(args.model_path,model_t,args)
  model_t.freeze_arch_parameters()
  # 冻结教师网络
  for para in list(model_t.parameters())[:-2]:
      para.requires_grad = False

  model_s = Network(C=args.init_channels,num_classes=args.class_num,layers=args.layers,steps=args.nodes,multiplier=args.nodes,
    stem_multiplier=args.stem_multiplier,group=args.group)
  model_s,_,_ = loadCheckpoint(args.model_path,model_s,args)
  model_s._initialize_alphas()

  criterion = nn.CrossEntropyLoss().to(args.device)
  model_d = Discriminator().to(args.device)
  model_s = model_s.to(args.device)
  logger.info("param size = %fMB", tools.count_parameters_in_MB(model_s))


  optimizer_d = optim.SGD(model_d.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  optimizer_s = optim.SGD(model_s.weight_parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  optimizer_m = FISTA(model_s.arch_parameters(), lr=args.learning_rate, gamma=args.sparse_lambda)

  scheduler_d = StepLR(optimizer_d, step_size=args.lr_decay_step, gamma=0.1)
  scheduler_s = StepLR(optimizer_s, step_size=args.lr_decay_step, gamma=0.1)
  scheduler_m = StepLR(optimizer_m, step_size=args.lr_decay_step, gamma=0.1)

  perf_scoreboard = PerformanceScoreboard(args.num_best_scores)


  if args.resume:
    logger.info('=> Resuming from ckpt {}'.format(args.resume_path))
    ckpt = torch.load(args.resume_path, map_location=args.device)
    start_epoch = ckpt['epoch']
    model_s.load_state_dict(ckpt['state_dict_s'])
    model_d.load_state_dict(ckpt['state_dict_d'])
    optimizer_d.load_state_dict(ckpt['optimizer_d'])
    optimizer_s.load_state_dict(ckpt['optimizer_s'])
    optimizer_m.load_state_dict(ckpt['optimizer_m'])
    scheduler_d.load_state_dict(ckpt['scheduler_d'])
    scheduler_s.load_state_dict(ckpt['scheduler_s'])
    scheduler_m.load_state_dict(ckpt['scheduler_m'])
    perf_scoreboard = ckpt['perf_scoreboard']
    logger.info('=> Continue from epoch {}...'.format(start_epoch))

  models = [model_t, model_s, model_d]
  optimizers = [optimizer_d, optimizer_s, optimizer_m]
  schedulers = [scheduler_d, scheduler_s, scheduler_m]

  for epoch in range(start_epoch, args.num_epochs):
    for s in schedulers:
      logger.info('epoch %d lr %e ', epoch, s.get_lr()[0])

    _, _, _ = train(trainLoader, models, epoch, optimizers, monitors, args,logger)
    v_top1, v_top5, v_loss = validate(valLoader, model_s,criterion,epoch, monitors, args,logger)

    l,board=perf_scoreboard.update(v_top1, v_top5, epoch)
    for idx in range(l):
        score = board[idx]
        logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    logger.info("normal: \n{}".format(model_s.alphas_normal.data.cpu().numpy()))
    logger.info("reduce: \n{}".format(model_s.alphas_reduce.data.cpu().numpy()))
    logger.info('Genotypev1: {}'.format(model_s.genotypev1()))
    logger.info('Genotypev2: {}'.format(model_s.genotypev2()))
    logger.info('Genotypev3: {}'.format(model_s.genotypev3()))
    mask = []
    pruned = 0
    num = 0
    for param in model_s.arch_parameters():
      weight_copy = param.clone()
      param_array = np.array(weight_copy.detach().cpu())
      pruned += sum(w == 0 for w in param_array)
      num += len(param_array)   
    logger.info("Epoch:{} Pruned {} / {}".format(epoch,pruned, num))

    if epoch% args.save_freq==0:
      model_state_dict = model_s.module.state_dict() if len(args.gpus) > 1 else model_s.state_dict()
      state = {
          'state_dict_s': model_state_dict,
          'state_dict_d': model_d.state_dict(),
          'optimizer_d': optimizer_d.state_dict(),
          'optimizer_s': optimizer_s.state_dict(),
          'optimizer_m': optimizer_m.state_dict(),
          'scheduler_d': scheduler_d.state_dict(),
          'scheduler_s': scheduler_s.state_dict(),
          'scheduler_m': scheduler_m.state_dict(),
          "perf_scoreboard" : perf_scoreboard,
          'epoch': epoch + 1
      }
      tools.save_model(state, epoch + 1, is_best,path=os.path.join(args.save,"ckpt"))
    # update learning rate
    for s in schedulers:
        s.step(epoch)



def train(trainLoader, models, epoch, optimizers, monitors, args,logger):
    device = args.device
    losses_d = AverageMeter()
    losses_data = AverageMeter()
    losses_g = AverageMeter()
    losses_sparse = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_t = models[0]
    model_s = models[1]
    model_d = models[2]

    bce_logits = nn.BCEWithLogitsLoss()

    optimizer_d = optimizers[0]
    optimizer_s = optimizers[1]
    optimizer_m = optimizers[2]

    # switch to train mode
    model_d.train()
    model_s.train()
        
    num_iterations = len(trainLoader)

    real_label = 1
    fake_label = 0

    for i, (inputs, targets) in enumerate(trainLoader, 1):
        num_iters = num_iterations * epoch + i
        inputs = inputs.to(device)
        targets = targets.to(device)
        features_t = model_t(inputs)
        features_s = model_s(inputs)
        ############################
        # (1) Update D network
        ###########################
        for p in model_d.parameters():  
            p.requires_grad = True  
        optimizer_d.zero_grad()
        output_t = model_d(features_t.detach())
        # print('output_t',output_t)
        labels_real = torch.full_like(output_t, real_label, device=device)
        error_real = bce_logits(output_t, labels_real)
        output_s = model_d(features_s.to(device).detach())
        # print('output_s',output_t)
        labels_fake = torch.full_like(output_s, fake_label, device=device)
        error_fake = bce_logits(output_s, labels_fake)

        error_d = error_real + error_fake

        # 对偶正则化，用来削弱鉴别其的收敛防止其过拟合
        labels = torch.full_like(output_s, real_label, device=device)
        error_d += bce_logits(output_s, labels)

        error_d.backward()
        losses_d.update(error_d.item(), inputs.size(0))
        optimizer_d.step()

        ############################
        # (2) Update student network
        ###########################
        for p in model_d.parameters():  
            p.requires_grad = False  

        optimizer_s.zero_grad()
        optimizer_m.zero_grad()

        error_data = args.miu * F.mse_loss(features_t, features_s.to(device))
        losses_data.update(error_data.item(), inputs.size(0))
        error_data.backward(retain_graph=True)

        # fool discriminator
        output_s = model_d(features_s.to(device))
        labels = torch.full_like(output_s, real_label, device=device)
        error_g = bce_logits(output_s, labels)
        losses_g.update(error_g.item(), inputs.size(0))
        error_g.backward(retain_graph=True)


        # train mask
        mask = []
        for name, param in model_s.named_parameters():
            if 'mask' in name:
                mask.append(param.view(-1))
        mask = torch.cat(mask)
        error_sparse = args.sparse_lambda * torch.norm(mask, 1)
        error_sparse.backward()
        losses_sparse.update(error_sparse.item(), inputs.size(0))
        optimizer_s.step()
        decay = (epoch % args.lr_decay_step == 0 and i == 1)
        if i % args.mask_step == 0:
            optimizer_m.step(decay)

        acc1, acc5 = accuracy(features_s.data, targets.data, topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()


        if (i + 1) % args.print_freq == 0:
            for m in monitors:
                m.update(epoch, i + 1, num_iterations, 'Training', {
                    'loss_discriminator': losses_d,
                    "loss_data": losses_data,
                    "loss_g": losses_g,
                    "loss_sparse": losses_sparse,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr']
                })
    logger.info('==> Top1: %.3f Top5: %.3f\n dis: %.3f data: %.3f g: %.3f spar:%.3f',
                top1.avg, top5.avg, losses_ds.avg, losses_data.avg,losses_g.avg,losses_sparse.avg)

    return top1.avg, top5.avg, None



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

