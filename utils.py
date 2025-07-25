import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

class Trainer:

    def __init__(self, trainloader, testloader, model, args, mode=None):
        
        self.trainloader = trainloader
        self.testloader = testloader

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(self.device)
        
        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model)
            cudnn.benchmark = True

        if mode == 'Train':
            # Optimizer
            self.opt = torch.optim.SGD(self.model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
            # Loss function
            self.criterion = nn.CrossEntropyLoss()
            # Learning rate schedule: piecewise decay
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [args.lr_chechpoint1, args.lr_chechpoint2], 0.1)

        self.args = args

    def pgd_attack(self, x_natural, y):
        x = x_natural.detach()   
        x = x + torch.zeros_like(x).uniform_(-self.args.epsilon, self.args.epsilon)   
        for _ in range(self.args.attack_iters):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.args.pgd_alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.args.epsilon), x_natural + self.args.epsilon)
          
        return x.detach()

    def train(self):
        self.model.train()   
        train_loss = 0
        train_acc = 0
        train_n = 0
        
        for _, (X, y) in enumerate(self.trainloader):
            X, y = X.to(self.device), y.to(self.device)
            self.opt.zero_grad()

            if self.args.train_mode == 'adv_train':
                X = self.pgd_attack(X, y)
            
            output = self.model(X)
            loss = self.criterion(output, y)
            assert not torch.isinf(loss).any()
            assert not torch.isnan(loss).any()
            loss.backward()
            self.opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        
        return train_loss/train_n, train_acc/train_n


    def evaluate(self, is_adv=False):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        test_n = 0
        for _, (X, y) in enumerate(self.testloader):
            X, y = X.to(self.device), y.to(self.device)
            if is_adv:
                X = self.pgd_attack(X, y)
            
            with torch.no_grad():
                output = self.model(X)
                loss = self.criterion(output, y)
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)
        return test_loss/test_n, test_acc/test_n




import logging
import os

def create_logger(log_path):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    return logger
