import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn

import random
import os
import numpy as np
import sys
import logging
import math
import utils
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


class FullConnect(nn.Module):
    def __init__(self,dimslist,numclass):
        super(FullConnect, self).__init__()

        # input se block
        inputdim = dimslist[0]
        self.se_fc1 = nn.Linear(inputdim,inputdim//4)
        self.se_fc2 = nn.Linear(inputdim//4,inputdim)

        # fc1--fc2--fc3(output)
        self.fc1 = nn.Linear(inputdim,dimslist[1])
        self.fc2 = nn.Linear(dimslist[1],dimslist[2])
        self.fc3 = nn.Linear(dimslist[2],numclass)

    def forward(self, input):
        w = F.relu(self.se_fc1(input))
        w = torch.sigmoid(self.se_fc2(w))
        out = input * w
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out,w




def cleanDate(path1,path2):
    if not (os.path.exists(path1) and os.path.exists(path1)):
        ValueError("input file is not exist !")
    havesample = []
    with open(path1,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.replace("\t",",")
            # 1,39,40,48及以后的列去除. 处理的时候决定48之后不保存，1 39 40 做置零处理
            # 只保存前47列
            line = line.split(",")[:47]
            line = list(map(int,line))
            # # 置零
            # line[0] = 0
            # line[38] = 0
            # line[39] = 0
            newline = []
            for i in range(len(line)):
                if i not in [0,38,39]:
                    newline.append(line[i])
            havesample.append(newline)
    # 正样本 8665条
    print(len(havesample))

    nosample = []
    with open(path2,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.replace("\t",",")
            # 1,39,40,48及以后的列去除. 处理的时候决定48之后不保存，1 39 40 做置零处理
            # 只保存前47列
            line = line.split(",")[:47]
            line = list(map(int,line))
            # 置零
            # line[0] = 0
            # line[38] = 0
            # line[39] = 0
            newline = []
            for i in range(len(line)):
                if i not in [0,38,39]:
                    newline.append(line[i])
            nosample.append(newline)
    # 负样本 3894条
    print(len(nosample))
    return havesample,nosample



class SamplesLoader(Dataset):
    def __init__(self,normalize,path1,path2):
        super(SamplesLoader, self).__init__()
        self.normalize = normalize
        # list[list[...]]
        self.havesamples,self.nosamples = cleanDate(path1,path2)
        self.le1,self.le2 = len(self.havesamples),len(self.nosamples)
        self.len = self.le1 + self.le2
        self.labels = [1 for i in range(self.le1)] + \
                      [0 for j in range(self.le2)]
        self.samples = self.havesamples + self.nosamples
        assert  len(self.samples) == len(self.labels)
        # # labels to one-hot   0:[1,0] 1:[0,1]
        # self.labels = list(np.eye(2)[self.labels])



    def _normalize(self,data,type=1):
        # Normalize by row  0
        if type == 0:
            dmax, dmin = np.max(data,axis=1,keepdims=True),np.min(data,axis=1,keepdims=True)
        elif type == 1:
            # Normalize by col  1
            dmax, dmin = np.max(data,axis=0,keepdims=True),np.min(data,axis=0,keepdims=True)
            dmean,dstd = np.mean(data,axis=0,keepdims=True),np.std(data,axis=0,keepdims=True)
        elif type == 2:
            # Normalize by global 2
            dmax, dmin = np.max(data),np.min(data)
        else:
            return data
        #data = (data - dmin) / (dmax - dmin)
        data = (data - dmean) / (dstd)
        data[np.isnan(data)] = 0
        return data


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # to ndarray
        self.samples = np.asarray(self.samples,np.float32)
        self.labels = np.asarray(self.labels,np.float32)

        # shuffle
        lenlist = [i for i in range(self.len)]
        np.random.shuffle(lenlist)
        self.samples = self.samples[lenlist]
        self.labels = self.labels[lenlist]

        # normalize
        self.samples = self._normalize(self.samples,type=self.normalize)

        # ndarray --> torch tensor
        self.samples= torch.from_numpy(self.samples)
        self.labels=torch.from_numpy(self.labels).long()
        return self.samples[index],self.labels[index]



def create_pic(data):

    x = [i+1 for i in range(len(data))]
    y = data
    plt.bar(x, y, color='g')

    plt.xlabel('dim index ')
    plt.ylabel('importance value')
    plt.xticks(x)
    plt.title('sigmod improtance')
    plt.grid(linestyle='-.')
    plt.legend()
    plt.savefig("./original.png")
    plt.close()


    data_exp = np.exp(np.asarray(data))
    soft_data = data_exp / np.sum(data_exp)
    y = list(soft_data)
    x = [i + 1 for i in range(len(y))]
    plt.bar(x, y, color='g')
    plt.xlabel('dim index ')
    plt.ylabel('importance value')
    plt.xticks(x)
    plt.title('sigmod improtance')
    plt.grid(linestyle='-.')
    plt.legend()
    plt.savefig("./softmax.png")
    plt.close()


def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))


def get_logger(log_dir):
    create_exp_dir(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'info.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('Train informations')
    logger.addHandler(fh)
    return logger



def main():
    if not torch.cuda.is_available():
        print('No GPU device available')


    path1 = "./HaveSample.txt"
    path2 = "./NoSample.txt"
    save = "./"

    manualSeed = 100
    gpus = 0
    normalize_type = 1
    train_portion = 0.8

    batch_size = 64
    epochs = 100
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    soptimizer = "sgd"
    sscheduler = "cos"
    steplist = [30,60]
    step_size = 40

    dimslist = [44,20,10]
    numclass = 2


    logger = get_logger("./")

    manualSeed = random.randint(1, 10000)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    use_cuda= gpus>0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.manual_seed(manualSeed)
        cudnn.benchmark = True

    model = FullConnect(dimslist,numclass)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    if torch.cuda.device_count() > 1 and use_cuda:
        logger.info('use: %d gpus', torch.cuda.device_count())
        model = nn.DataParallel(model)
    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)


    # loader
    loader = SamplesLoader(normalize_type,path1,path2)
    num_train = len(loader)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))
    train_queue = torch.utils.data.DataLoader(
        loader, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        loader, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)
    # lr_scheduler
    if soptimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    elif soptimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     betas=(0.9, 0.999),
                                     weight_decay=weight_decay)
    else:
        NotImplementedError()

    if sscheduler.lower() == 'warm_up_cos':
        warm_up_epochs = 5
        warm_up_with_adam = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
            else 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_adam)
    elif sscheduler.lower() == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)
    elif sscheduler.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.1,
                                                    last_epoch=-1)
    elif sscheduler.lower() == "mstep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steplist, gamma=0.1)
    else:
        NotImplementedError()
        return

    best_acc = 0.0
    for epoch in range(epochs):
        logger.info('Epoch: {}  lr: {}'.format(epoch, scheduler.get_lr()[0]))
        train_acc, train_obj = train(train_queue, model, criterion, optimizer,device)
        logger.info('Train_acc: %f', train_acc)
        valid_acc, valid_obj, avg_sigmod = infer(valid_queue, model, criterion,device)
        logger.info('Train_acc: {} Valid_acc: {}'.format(train_acc, valid_acc))
        logger.info("Avg sigmod: {}".format(avg_sigmod))
        if valid_acc > best_acc:
            best_acc = valid_acc
            utils.save(model, os.path.join(save, 'best_weights.pt'))
        scheduler.step()



    valid_acc, valid_obj, avg_sigmod = infer(valid_queue, model, criterion, device)
    logger.info("final model sigmod: {}".format(avg_sigmod))

    model.load_state_dict(torch.load(os.path.join(save, 'best_weights.pt'),
                                     map_location=device))
    valid_acc, valid_obj, avg_sigmod = infer(valid_queue, model, criterion, device)
    logger.info("best model sigmod: {}".format(avg_sigmod))
    create_pic(avg_sigmod)



def train(train_queue, model, criterion, optimizer,device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits, w = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        prec1 = utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1[0].data.item(), n)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion,device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    # Mean of importance
    counter = 0
    total_sigmod = 0
    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)

        with torch.no_grad():
            logits, w = model(input)
            loss = criterion(logits, target)
            if counter ==0:
                total_sigmod = np.mean(w.cpu().numpy(),axis=0,keepdims=False)
            else:
                total_sigmod += np.mean(w.cpu().numpy(), axis=0, keepdims=False)
            counter += 1
        prec1= utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1[0].data.item(), n)

    # batch_size, inputdim
    avg_sigmod = total_sigmod / counter
    return top1.avg, objs.avg, list(avg_sigmod)




if __name__=="__main__":
    # path1 = "./HaveSample.txt"
    # path2 = "./NoSample.txt"
    # havesamples,nosamples=cleanDate(path1,path2)
    # testDatanormalize(havesamples,nosamples)

    main()


    # final_sigmod =







