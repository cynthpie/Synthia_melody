import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module import RevGrad
import os, sys
from torch.autograd import Function
import argparse

class Resblock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, drop_out_rate=0.2, kernel_size=3, padding=1, is_discriminator=False):
        super(Resblock, self).__init__()
        if is_discriminator:
            self.left = nn.Sequential(
                nn.Dropout(p=drop_out_rate), # add drop out to encoded feature
                nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm1d(outchannel),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(outchannel)
            )
        else:
            self.left = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm1d(outchannel),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=drop_out_rate),
                nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(outchannel)
            )
        self.shortcut=nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            if kernel_size ==3 and stride==1:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm1d(outchannel)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm1d(outchannel)
                )

        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.leaky_relu(out)
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
            nn.LeakyReLU()
        )
        self.layer1 = self.make_layer(Resblock, args.foc*2, args.num_block_e, stride = 1)
        # self.layer2 = self.make_layer(Resblock, args.foc*4, 5, stride = 1) # reduce layer here
        self.conv2 = nn.Conv1d(args.foc*4, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.remove_classifier = args.remove_classifier

    def make_layer(self, block, channels, num_blocks, stride):
        if num_blocks == 0:
            return nn.Sequential()
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

        self.layer1 = self.make_layer(Resblock, args.foc*4, args.num_blocks_c2, stride = 1, drop_out_rate=args.drop_out_rate_c)
        self.layer2 = self.make_layer(Resblock, args.foc*2, args.num_blocks_c, stride = 1, drop_out_rate=args.drop_out_rate_c)
        self.classifier = nn.Sequential(
            nn.Conv1d(self.inchannel, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def make_layer(self, block, channels, num_blocks, stride, drop_out_rate):
        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, drop_out_rate))
            self.inchannel = channels #match output channel to next input channel 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
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
        
        self.layer1 = self.make_layer(Resblock, args.foc*4, args.num_blocks_d2, stride = args.stride, drop_out_rate=args.drop_out_rate_d,\
            kernel_size=args.kernel_size, is_discriminator=True)
        self.layer2 = self.make_layer(Resblock, args.foc*2, args.num_blocks_d, stride = args.stride, drop_out_rate=args.drop_out_rate_d,\
            kernel_size=args.kernel_size, is_discriminator=False)
        self.discriminator = nn.Sequential(
            # nn.Conv1d(self.inchannel, self.inchannel//2, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm1d(self.inchannel//2),
            # nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.inchannel, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def make_layer(self, block, channels, num_blocks, stride, drop_out_rate, kernel_size, is_discriminator):
        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        same_padding = 1
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, drop_out_rate, kernel_size, same_padding, is_discriminator=True))
            self.inchannel = channels #match output channel to next input channel 
        return nn.Sequential(*layers)
    
    def forward(self, x, alpha):
        # x = ReverseLayerF.apply(x, alpha)
        x = self.layer1(x)
        x = self.layer2(x)
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
            output = grad_output.neg() * ctx.alpha # remove negative here
        return output, None

def get_discriminator_args():
    parser = argparse.ArgumentParser(description='discriminator_args')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv1D in SampleCNN [default: 64]')
    parser.add_argument("--num_domain", type=int, default=2,
                        help="number of domain in training and testing data [default: 2]")
    parser.add_argument("--drop_out_rate_d", type=float, default=0.85,
                        help="number of class labels [default: 2]")
    parser.add_argument("--stronger_dis", type=bool, default=False,
                        help="number of class labels [default: 2]")
    parser.add_argument("--num_blocks_d", type=bool, default=1,
                        help="number of class labels [default: 2]")
    parser.add_argument("--num_blocks_d2", type=bool, default=1,
                        help="number of class labels [default: 2]")
    parser.add_argument("--kernel_size", type=bool, default=22,
                        help="number of class labels [default: 2]")
    parser.add_argument("--stride", type=bool, default=22,
                        help="number of class labels [default: 2]")
    args = parser.parse_args()
    return args

def get_extractor_args():
    parser = argparse.ArgumentParser(description='extractor_args')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv1D in SampleCNN [default: 64]')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of class labels [default: 2]")
    parser.add_argument("--remove_classifier", type=bool, default=True,
                        help="remove last convolution layer in the model for DANN [default: 2]")
    parser.add_argument("--num_block_e", type=int, default=1,
                        help="remove last convolution layer in the model for DANN [default: 2]")
    args = parser.parse_args()
    return args

def get_classifier_args():
    parser = argparse.ArgumentParser(description='extractor_args')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv1D in SampleCNN [default: 64]')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of class labels [default: 2]")
    parser.add_argument("--drop_out_rate_c", type=float, default=0.2,
                        help="number of class labels [default: 2]")
    parser.add_argument("--stronger_clas", type=bool, default=False,
                        help="number of class labels [default: 2]")
    parser.add_argument("--num_blocks_c", type=bool, default=4,
                        help="number of class labels [default: 2]")
    parser.add_argument("--num_blocks_c2", type=bool, default=4,
                        help="number of class labels [default: 2]")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    batch_size = 3
    data = torch.rand(batch_size ,1, 64323)
    args_d= get_discriminator_args()
    args_e= get_extractor_args()
    args_c = get_classifier_args()
    e = FeatureExtractor(args_e)
    d = Discriminator(args_d)
    c = Classifier(args_c)
    f = e(data)
    print(f.size())
    x = d(f, 3)
    print(x.size())
    x = c(f)
    print(x.size())

