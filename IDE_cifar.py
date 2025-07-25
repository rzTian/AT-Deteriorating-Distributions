import argparse
import logging
import sys
import time
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Induced Distribution Experiment (IDE)')
parser.add_argument('--dataset', type=str,   default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--load_chkpt', default=200, type=int) 
parser.add_argument('--train_mode', type=str,   default='std_train', choices=['adv_train', 'std_train'])
# Hyper-parameters and network settings
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--model', type=str,   default='PreRes18_standard')
parser.add_argument('--lr_max', default=0.1, type=float)
parser.add_argument('--lr_schedule', default='piecewise', choices=['piecewise', 'cosine'])
parser.add_argument('--lr_chechpoint1', type=int, default=60)
parser.add_argument('--lr_chechpoint2', type=int, default=80)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
# Other settings
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

perturbed_data = './cifar100_perturbed' if (args.dataset == 'cifar100') else './cifar10_perturbed'
fname = 'IDE_results_cifar100' if (args.dataset == 'cifar100') else 'IDE_results_cifar10'
num_class = 100 if (args.dataset == 'cifar100') else 10


if not os.path.exists(fname):
        os.makedirs(fname)

# Create logger to track the training trajectory
log_path = os.path.join(fname, f'IDE{args.seed}_chkpt_{args.load_chkpt}.log')

from utils import create_logger
logger = create_logger(log_path)
logger.info(args)


#### Loading perturbed dataset ####
#### The data are originally saved from CUDA, should be loaded on cpu to avoid errors.
trainset = torch.load(os.path.join(perturbed_data, f'{args.model}{args.seed}_chkpt_{args.load_chkpt}_trainset'), map_location='cpu')
testset = torch.load(os.path.join(perturbed_data, f'{args.model}{args.seed}_chkpt_{args.load_chkpt}_testset'), map_location='cpu')

trainset = Data.TensorDataset(trainset['image'], trainset['labels'])
testset = Data.TensorDataset(testset['image'], testset['labels'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)


#### Loading model ####
# Models are all copied or modified from https://github.com/kuangliu/pytorch-cifar
if args.model == 'WideResNet':
    from models.wideresnet import *
    model = WideResNet(depth=34, num_classes=num_class, widen_factor=10, dropRate=0.0)
elif args.model ==  'PreRes18_standard':
    from models.Preact_ResNet_standard import *
    model = PreActResNet18(num_classes=num_class)
elif args.model == 'PreRes34_standard':
    from models.Preact_ResNet_standard import *
    model = PreActResNet34(num_classes=num_class)
else:
    raise ValueError

from utils import Trainer
IDE_trainer = Trainer(trainloader = trainloader,
                    testloader = testloader, 
                    model = model, 
                    args = args, 
                    mode = 'Train')



#########
logger.info('Epoch \t Train Time \t Test Time \t LR \t \t \t Train Loss \t Train Acc  \t Test Loss \t Test Acc')
for epoch in range(1, args.epochs):
    start_time = time.time()
    train_loss, train_acc = IDE_trainer.train()   # Set args.train_mode = 'std_train' to perform clean training
    train_time = time.time()
    IDE_trainer.scheduler.step()  # upgrade the learning rate after each epoch

    test_loss, test_acc = IDE_trainer.evaluate(is_adv = False)                                 
    test_time = time.time()  
   
    logger.info('%d \t\t %.1f \t \t %.1f \t \t %.4f \t\t\t %.4f \t %.4f \t %.4f \t %.4f',
        epoch, train_time - start_time, test_time - train_time, IDE_trainer.scheduler.get_last_lr()[0],
        train_loss, train_acc, test_loss, test_acc)
