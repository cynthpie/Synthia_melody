import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module import RevGrad
import os, sys
sys.path.append("/rds/general/user/cl222/home/msc_project/sampleCNN")
from sampleCNN import MySampleCNN

class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.extractor = MySampleCNN(args) 

    def forward(self, x):
        x = self.extractor(x)
        x = torch.logit(x) # since MySampleCNN return sigmoid score, revert back to raw logit
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        if args.num_domain==2:
            num_domain = 1
        else:
            num_domain = args.num_domain
        self.linear = nn.Linear(args.in_dim, num_domain) # logistic regression
        self.revgrad = RevGrad(alpha=args.alpha)
    
    def forward(self, x):
        x = x[:,None]
        x = self.revgrad(x)
        x = self.linear(x)
        x = torch.squeeze(x)
        return torch.sigmoid(x)
