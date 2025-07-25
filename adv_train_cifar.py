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
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms


###################################################
parser = argparse.ArgumentParser(description='Perform adversarial training and save checkpoints')
# Hyper-parameters and network settings
parser.add_argument('--dataset', type=str,   default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=201, type=int)
parser.add_argument('--model', type=str,   default='PreRes18_standard')
parser.add_argument('--lr_max', default=0.1, type=float)
parser.add_argument('--width_factor', default=10, type=int)
parser.add_argument('--lr_schedule', default='piecewise', choices=['piecewise', 'cosine'])
parser.add_argument('--lr_chechpoint1', type=int, default=100)
parser.add_argument('--lr_chechpoint2', type=int, default=150)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4) 
# Settings for adv training
parser.add_argument('--train_mode', type=str,   default='adv_train', choices=['adv_train', 'std_train'])
parser.add_argument('--epsilon', default=8/255., type=float) 
parser.add_argument('--attack_iters', default=10, type=int) 
parser.add_argument('--pgd_alpha', default=2/255., type=float) 
# Other settings
parser.add_argument('--seed', default=1, type=int)  
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--chkpt_iters', default=20, type=int)
args = parser.parse_args()
###################################################


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Setup folders to save model's weights
fname = 'cifar10_chkpts' if (args.dataset == 'cifar10') else 'cifar100_chkpts'
num_class = 100 if (args.dataset == 'cifar100') else 10
if not os.path.exists(fname):
        os.makedirs(fname)

# Create logger to track the training trajectory
log_path = os.path.join(fname, f'{args.model}{args.seed}.log')

from utils import create_logger
logger = create_logger(log_path)
logger.info(args)

#### Load data  ####
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),])

    trainset = torchvision.datasets.CIFAR100(root='./cifar100-data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./cifar100-data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


#### Loading model ####
if args.model == 'WideResNet':
    from models.wideresnet import *
    model = WideResNet(depth=34, num_classes=num_class, widen_factor=args.width_factor, dropRate=0.0)
elif args.model ==  'PreRes18_standard':
    from models.Preact_ResNet_standard import *
    model = PreActResNet18(num_classes=num_class)
elif args.model == 'PreRes34_standard':
    from models.Preact_ResNet_standard import *
    model = PreActResNet34(num_classes=num_class)
else:
    raise ValueError

#######
from utils import Trainer
AT_trainer = Trainer(trainloader = trainloader,
                    testloader = testloader, 
                    model = model, 
                    args = args, 
                    mode = 'Train')
#######
   
logger.info('Epoch \t Train Time \t Test Time \t LR \t \t \t Train Adv Loss \t Train Adv Acc \t Test Adv Loss \t Test Adv Acc ')
for epoch in range(1, args.epochs):
    start_time = time.time()
    train_robust_loss, train_robust_acc = AT_trainer.train() # Adversarial training
    train_time = time.time()
    AT_trainer.scheduler.step()  # upgrade the learning rate after each epoch
    test_robust_loss, test_robust_acc = AT_trainer.evaluate(is_adv = True)
    test_time = time.time()
        
    logger.info('%d \t\t %.1f \t \t %.1f \t %.4f \t \t \t %.4f \t %.4f \t %.4f \t %.4f',
        epoch, train_time - start_time, test_time - train_time, AT_trainer.scheduler.get_last_lr()[0],
        train_robust_loss, train_robust_acc,
        test_robust_loss, test_robust_acc)


    # save checkpoint
    if epoch % args.chkpt_iters == 0 or epoch == (args.epochs-1):
        torch.save(AT_trainer.model.state_dict(), os.path.join(fname, f'{args.model}{args.seed}_{epoch}.pth'))
