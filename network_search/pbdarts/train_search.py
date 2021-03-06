import os
import sys
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
import copy
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype
import yaml
import math

import tools
sys.path.append("../../")
from  utils import *


def get_args():
    parser = argparse.ArgumentParser("pdarts original cifar search")
    parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--print_freq', type=float, default=150, help='report frequency')
    parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=5, help='total number of layers')
    parser.add_argument('--nodes', type=int, default=4, help='middle nodes')
    parser.add_argument('--multiplier', type=int, default=4, help='middle nodes')
    parser.add_argument('--stem_multiplier', type=int, default=3, help='stem_multiplier')
    parser.add_argument('--gpus', type=int, default=1, help='gpus number')

    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment path')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--tmp_data_dir', type=str, default=r'F:\source_code\DataSets', help='temp data dir')
    parser.add_argument('--note', type=str, default='try', help='note for this run')
    parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
    parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
    parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
    parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')

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
        logger.info('no gpu device available')
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


    if args.cifar100:
        CIFAR_CLASSES = 100
        data_folder = 'cifar-100-python'
    else:
        CIFAR_CLASSES = 10
        data_folder = 'cifar-10-batches-py'

    #  prepare dataset
    if args.cifar100:
        train_transform, valid_transform = tools._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = tools._data_transforms_cifar10(args)

    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        vaild_ata = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=False, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        vaild_ata = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=False, transform=valid_transform)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)

    valLoader = torch.utils.data.DataLoader(
      vaild_ata, batch_size=args.batch_size,
      pin_memory=True, num_workers=args.workers)
    
    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 6, 12]
    if len(args.dropout_rate) ==3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.1, 0.4, 0.7]
    eps_no_archs = [10, 10, 10]
    state_epochs = 0
    for sp in range(len(num_to_keep)):
        model = Network(args.init_channels + int(add_width[sp]), CIFAR_CLASSES, args.layers + int(add_layers[sp]), criterion, 
            steps=args.nodes, multiplier=args.multiplier, stem_multiplier=args.stem_multiplier, 
            switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_rate[sp]))

        model = model.to(args.device)
        logger.info("stage:{} param size:{}MB".format(sp,tools.count_parameters_in_MB(model)))

        optimizer = torch.optim.SGD(
                model.weight_parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(
                    model.arch_parameters(),
                    lr=args.arch_learning_rate, 
                    betas=(0.5, 0.999), 
                    weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        for epoch in range(epochs):
            lr = scheduler.get_lr()[0]
            logger.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch:
                model.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.update_p()
                train_acc, train_obj = train(state_epochs+epoch,train_queue, valid_queue, model, criterion, optimizer, optimizer_a, 
                    args,monitors,logger, train_arch=False)
            else:
                model.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor) 
                model.update_p()                
                train_acc, train_obj = train(state_epochs+epoch,train_queue, valid_queue, model, criterion, optimizer, optimizer_a, 
                    args,monitors,logger, train_arch=True)

            # validation
            valid_acc, valid_obj = infer(state_epochs+epoch,valLoader, model, criterion,args,monitors,logger)

            if epoch >= eps_no_arch:
                # 将本epoch的解析结果保存
                arch_param = model.arch_parameters()
                normal_prob = F.softmax(arch_param[0], dim=-1).data.cpu().numpy()
                reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
                logger.info('Genotypev: {}'.format(parse_genotype(switches_normal.copy(),switches_reduce.copy(),
                                                          normal_prob.copy(),reduce_prob.copy())))

            scheduler.step()

        tools.save(model, os.path.join(args.save, 'state{}_weights.pt'.format(sp)))
        state_epochs += args.epochs

        # Save switches info for s-c refinement. 
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        arch_param = model.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=-1).data.cpu().numpy()
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()

        logger.info('------Stage %d end!------' % sp)
        logger.info("normal: \n{}".format(normal_prob))
        logger.info("reduce: \n{}".format(reduce_prob))
        logger.info('Genotypev: {}'.format(parse_genotype(switches_normal.copy(),switches_reduce.copy(),
                                                          normal_prob.copy(),reduce_prob.copy())))

        # 根据最新的结构权重,旧的搜索空间,需要抛弃的数量,当前状态 来进行空间正则化
        switches_normal = update_switches(normal_prob.copy(),switches_normal,
                                            num_to_drop[sp],sp,len(num_to_keep))
        switches_reduce = update_switches(reduce_prob.copy(),switches_reduce,
                                            num_to_drop[sp],sp,len(num_to_keep))

        logger.info('------Dropping %d paths------' % num_to_drop[sp])
        logger.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal,logger)
        logger.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce,logger)


        if sp == len(num_to_keep) - 1:
            # arch_param = model.arch_parameters()
            # normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            # reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])                
            # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logger.info(genotype)
            ## restrict skipconnect (normal cell only)
            logger.info('Restricting skipconnect...')
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks                
                num_sk = check_sk_number(switches_normal)               
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logger.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logger.info(genotype)              


def train(epoch,train_queue, valid_queue, model, criterion, optimizer, optimizer_a,
            args,monitors,logger, train_arch=True):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    total_sample = len(train_queue.sampler)
    batch_size = train_queue.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.to(args.device)
        target = target.to(args.device)
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above. 
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.to(args.device)
            target_search = target_search.to(args.device)
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if (step + 1) % args.print_freq == 0:
            for m in monitors:
                m.update(epoch, step + 1, steps_per_epoch, 'Training', {
                  'Loss': objs,
                  'Top1': top1,
                  'Top5': top5,
                  'LR': optimizer.param_groups[0]['lr']
              })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                    top1.avg, top5.avg, objs.avg)

    return top1.avg, objs.avg


def infer(epoch,valid_queue, model, criterion,args,monitors,logger):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    total_sample = len(valid_queue.sampler)
    batch_size = valid_queue.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)


    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = input.to(args.device)
        target = target.to(args.device)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)


        if (step + 1) % args.print_freq == 0:
            for m in monitors:
                m.update(epoch, step + 1, steps_per_epoch, 'Validation', {
                  'Loss': objs,
                  'Top1': top1,
                  'Top5': top5,
              })
    
    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                    top1.avg, top5.avg, objs.avg)

    return top1.avg, objs.avg



def update_switches(switches_weights,old_switches,drop_num,state,total_state=3):
    new_switches = old_switches.copy()
    assert len(switches_weights) == len(old_switches) and state< total_state
    for i in range(len(old_switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if old_switches[i][j]:
                idxs.append(j)
        assert len(idxs) == len(list(switches_weights[i]))
        if state == total_state - 1:
            drop = get_min_k_no_zero(switches_weights[i, :], idxs, drop_num)
        else:
            drop = get_min_k(switches_weights[i, :], drop_num)
        for idx in drop:
            new_switches[i][idxs[idx]] = False
    return new_switches

def parse_genotype(switches_normal,switches_reduce,normal_weight,reduce_weight):

    def parse_gen(switches,weights):
        mixops = []
        assert len(switches) == len(weights)
        for i in range(len(switches)):
            mixop_array = weights[i]
            keep_idx = []
            for j in range(len(PRIMITIVES)):
                if switches[i][j]:
                    keep_idx.append(j)
            assert len(keep_idx) == len(mixop_array)
            if switches[i][0]:
                mixop_array[0]=0
            max_value, max_index = float(np.max(mixop_array)), int(np.argmax(mixop_array))
            max_index_pri = keep_idx[max_index]
            max_op_name = PRIMITIVES[max_index_pri]
            assert max_op_name!='none'
            mixops.append((max_value,max_op_name))
        # get the final cell genotype based in normal_down_res
        n = 2
        start = 0
        mixops_gen=[]
        for i in range(4):
            end=start+n
            node_egdes=mixops[start:end].copy()
            keep_edges=sorted(range(2 + i), key=lambda x: -node_egdes[x][0])[:2]
            for j in keep_edges:
                op_name=node_egdes[j][1]
                mixops_gen.append((op_name,j))
            start=end
            n+=1
        return mixops_gen

    normal_gen = parse_gen(switches_normal,normal_weight)
    reduce_gen = parse_gen(switches_reduce,reduce_weight)
    concat = range(2,6)
    genotype = Genotype(
      normal=normal_gen, normal_concat=concat,
      reduce=reduce_gen, reduce_concat=concat
    )
    return genotype


def parse_network(switches_normal, switches_reduce):
    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)
    concat = range(2, 6)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat, 
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1
    
    return index
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True 
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index
        
def logging_switches(switches,logger):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logger.info(ops)
        
def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1
    
    return count

def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0
    
    return probs_out

def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False            
    return switches

def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False  
    return switches  

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logger.info('Total searching time: %ds', duration)
