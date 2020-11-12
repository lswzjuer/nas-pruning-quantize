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
from binary_model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("search")
parser.add_argument('--dataset_name', type=str, default='cifar10', help='dataset name')
parser.add_argument('--class_num', type=int, default=10, help='num of dataset class')
parser.add_argument('--dataset', type=str, default='/share/jjchu/source_code/DataSets', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--group', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/share/jjchu/source_code/checkpoints/darts', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--gpus',type=int,default=1,help="gpu number")
args = parser.parse_args()


args.save = '{}/search-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  #torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)
  args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
  args.device = torch.device('cuda:0' if args.use_cuda else 'cpu')


  criterion = nn.CrossEntropyLoss()
  model = Network(args.init_channels, args.class_num, args.layers, criterion,group=args.group)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


  if args.gpus>1:
      logging.info('use: %d gpus', args.gpus)
      model = nn.DataParallel(model)
  criterion = criterion.to(args.device)
  model = model.to(args.device)

  # get dataloader
  if args.dataset_name == "cifar10":
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    traindata = dset.CIFAR10(root=args.dataset, train=True, download=False, transform=train_transform)
    valdata = dset.CIFAR10(root=args.dataset, train=False, download=False, transform=valid_transform)
  else:
    train_transform, valid_transform = utils._data_transforms_mnist(args)
    traindata = dset.MNIST(root=args.dataset, train=True, download=False, transform=train_transform)
    valdata = dset.MNIST(root=args.dataset, train=False, download=False, transform=valid_transform)

  num_train = len(traindata)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))
  train_queue = torch.utils.data.DataLoader(
      traindata, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)
  valid_queue = torch.utils.data.DataLoader(
      traindata, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  if args.gpus > 1:
    optimizer = torch.optim.SGD(
        model.module.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    architect = Architect(model.module, args)
  else:
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    architect = Architect(model, args)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.module.genotype() if args.gpus > 1 else model.genotype()
    logging.info('genotype = %s', genotype)

    if args.gpus > 1:
      print(F.softmax(model.module.alphas_normal, dim=-1))
      print(F.softmax(model.module.alphas_reduce, dim=-1))
    else:
      print(F.softmax(model.alphas_normal, dim=-1))
      print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,device=args.device)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion,device=args.device)
    logging.info('valid_acc %f', valid_acc)
    
    model_state_dict = model.module.state_dict() if args.gpus > 1 else model.state_dict()
    torch.save(model_state_dict, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()
  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    input = Variable(input, requires_grad=False).to(device)
    target = Variable(target, requires_grad=False).to(device)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).to(device)
    target_search = Variable(target_search, requires_grad=False).to(device)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    
    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.to(device)
      target = target.to(device)
      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

