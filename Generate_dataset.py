import argparse
import sys
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

parser = argparse.ArgumentParser(description='Generate perturbed datasets for IDEs')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--width_factor', default=10, type=int)
# Related to the checkpoint saved file name
parser.add_argument('--model', type=str,   default='PreRes18_standard')
parser.add_argument('--seed_id', default=1, type=int) # the seed id of the trained model 
parser.add_argument('--seed', default=1, type=int) # random seed

parser.add_argument('--epsilon', default=8/255., type=float) 
parser.add_argument('--attack_iters', default=10, type=int) 
parser.add_argument('--pgd_alpha', default=2/255., type=float)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


fname = 'cifar10_chkpts' if (args.dataset == 'cifar10') else 'cifar100_chkpts' # The folder where AT checkpoints are saved.
dataDir_perturb = './cifar100_perturbed' if (args.dataset == 'cifar100') else './cifar10_perturbed' # The folder to save the perturbed datasets.
num_class = 100 if (args.dataset == 'cifar100') else 10

if not os.path.exists(dataDir_perturb):
        os.makedirs(dataDir_perturb)


if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)  
    testset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=False, download=True, transform=transforms.ToTensor()) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./cifar100-data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./cifar100-data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
else:
    raise ValueError

#### Loading model ####
# Models are all copied or modified from https://github.com/kuangliu/pytorch-cifar
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

#######################

from utils import Trainer
data_generator = Trainer(trainloader = trainloader,
                    testloader = testloader, 
                    model = model, 
                    args = args, 
                    mode = None)

#### Add new method to Trainer for generating perturbed datasets

def generate_data(self, is_trainset = True):
    self.model.eval()
    new_data = []
    label = []
    acc = 0
    num_data = 0
    loader = self.trainloader if is_trainset else self.testloader
    for batch_id, (X, y) in enumerate(loader):
        X, y = X.to(self.device), y.to(self.device)
        X = self.pgd_attack(X, y)
        new_data.append(X)
        label.append(y)
        # sanity check purpose
        with torch.no_grad():
            output = self.model(X)
            acc += (output.max(1)[1] == y).sum().item()
            num_data += y.size(0)

    return torch.cat(new_data, dim=0), torch.cat(label), acc/num_data

from types import MethodType
data_generator.generate_data = MethodType(generate_data, data_generator)

######################
checkpoints = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

for chechpoint_id in checkpoints:
    if data_generator.device == 'cuda':
       data_generator.model.load_state_dict(torch.load(os.path.join(fname, f'{args.model}{args.seed_id}_{chechpoint_id}.pth')))
    else:
        state_dict = torch.load(os.path.join(fname, f'{args.model}{args.seed_id}_{chechpoint_id}.pth'), map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        data_generator.model.load_state_dict(new_state_dict)
    
    print("Generating dataset ", chechpoint_id)
    Img_train, label_train, train_acc = data_generator.generate_data(is_trainset=True)
    
    print("Number of training images:", Img_train.size())
    print("Number of labels:", label_train.size())
    print("train robust acc of the loaded chkpt:", train_acc)

    Img_test, label_test, test_acc = data_generator.generate_data(is_trainset=False)
    
    print("Number of testing images:", Img_test.size())
    print("Number of labels:", label_test.size())
    print("test robust acc of the loaded chkpt:", test_acc)
    

    torch.save({'image':Img_train, 'labels':label_train}, os.path.join(dataDir_perturb, f'{args.model}{args.seed_id}_chkpt_{chechpoint_id}_trainset'))
    torch.save({'image':Img_test, 'labels':label_test}, os.path.join(dataDir_perturb, f'{args.model}{args.seed_id}_chkpt_{chechpoint_id}_testset'))

print("Done!")