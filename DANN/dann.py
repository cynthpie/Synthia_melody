import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module import RevGrad
import os, sys
from torch.autograd import Function

class Resblock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Resblock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut=nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(outchannel)
            )
        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.maxpool(out)
        return out

# define sampleCNN
class SampleCNN(nn.Module):
    def __init__(self, Resblock, args):
        super(SampleCNN, self).__init__()
        if args.num_classes==2:
            num_classes = 1
        else:
            num_classes = args.num_classes
        self.inchannel = args.foc
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, args.foc, kernel_size = 3, stride = 3, padding = 1, bias = False), 
            nn.BatchNorm1d(args.foc), 
            nn.ReLU()
        )
        self.layer1 = self.make_layer(Resblock, args.foc*2, 5, stride = 1)
        # self.layer2 = self.make_layer(Resblock, args.foc*4, 1, stride = 1) # reduce layer here
        self.conv2 = nn.Conv1d(args.foc*4, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.remove_classifier = args.remove_classifier

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels #match output channel to next input channel 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # input x dim: batch_size x 1 x 64323
        x = self.conv1(x) # x = batch_size x args.foc x 21441
        x = self.layer1(x) # x = batch_size x args.foc*2 x 88
        # x = self.layer2(x) # x = batch_size x args.foc*4 x 1
        if not self.remove_classifier: # for dann
            x = self.conv2(x)  # x = batch_size x 1 x 1
            return torch.sigmoid(x)
        return x

def MySampleCNN(args):
    return SampleCNN(Resblock, args)

class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.extractor = MySampleCNN(args) 

    def forward(self, x):
        x = self.extractor(x)
        return x

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.inchannel = args.foc*2
        if args.num_classes==2:
            num_classes = 1
        else:
            num_classes = args.num_classes
        if args.stronger_clas:
            self.classifier = nn.Sequential(
                nn.Conv1d(args.foc*4, args.foc*2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(args.foc*2),
                nn.Dropout(p=args.drop_out_rate_c),
                nn.Conv1d(args.foc*2, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Conv1d(args.foc*4, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.layer1 = self.make_layer(Resblock, args.foc*4, args.num_blocks_c, stride = 1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels #match output channel to next input channel 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.inchannel = args.foc*2
        if args.num_domain==2:
            num_domain = 1
        else:
            num_domain = args.num_domain
        
        if args.stronger_dis:
            self.discriminator = nn.Sequential(
                nn.Dropout(p=args.drop_out_rate_d),
                nn.Conv1d(args.foc*4, args.foc*2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(args.foc*2),
                nn.ReLU(inplace=True),
                nn.Conv1d(args.foc*2, num_domain, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Dropout(p=args.drop_out_rate_d),
                nn.Conv1d(args.foc*4, num_domain, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.layer1 = self.make_layer(Resblock, args.foc*4, args.num_blocks_d, stride = 1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels #match output channel to next input channel 
        return nn.Sequential(*layers)
    
    def forward(self, x, alpha):
        reversed_input = ReverseLayerF.apply(x, alpha)
        x = self.layer1(x)
        x = self.discriminator(x)
        return torch.sigmoid(x)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            output = grad_output * ctx.alpha # remove negative here
        return output, None
