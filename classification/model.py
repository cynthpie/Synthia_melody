# define resnet building blocks
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1):    
        super(ResidualBlock, self).__init__() 
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel: 
            # 1x1 conv act as dimension adaptor
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 
                                                 kernel_size=1, stride=stride, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel) ) 
    
    def forward(self, x): 
        out = self.left(x) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        return out

    
# define resnet

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, args, num_classes = 24):
        super(ResNet, self).__init__()
        self.inchannel = args.foc
        self.conv1 = nn.Sequential(nn.Conv2d(1, args.foc, kernel_size = 3, stride = 1, #
                                            padding = 1, bias = False), 
                                  nn.BatchNorm2d(args.foc), #
                                  nn.ReLU())
        
        self.layer1 = self.make_layer(ResidualBlock, args.foc*2, 2, stride = (2,4))
        self.layer2 = self.make_layer(ResidualBlock, args.foc*4, 2, stride = (2,4))
        self.layer3 = self.make_layer(ResidualBlock, args.foc*8, 2, stride = (2,4))
        self.layer4 = self.make_layer(ResidualBlock, args.foc*16, 2, stride = (2,4))
        self.maxpool = nn.MaxPool2d((2,4))
        # self.fc = nn.Linear(args.foc*16, num_classes) 
        self.conv2 = nn.Conv2d(args.foc*16, num_classes, kernel_size = 1, stride = 1, #
                                            padding = 0, bias = False)
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels #match output channel to next input channel 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x_dim = 1x32x887
        x = self.conv1(x) #x_dim = args.foc x 32 x 887
        x = self.layer1(x)  #x_dim = args.foc*2 x 16 x 222
        x = self.layer2(x) #x_dim = args.foc*4 x 8 x 56
        x = self.layer3(x) #x_dim = args.foc*8 x 4 x 14
        x = self.layer4(x) #x_dim = args.foc*16 x 2 x 4
        x = self.maxpool(x) #x_dim = args.foc*16 x 1 x 1
        x = self.conv2(x)  # x_dim = batch_size x nb_class x 1 x1 
        return x
    
# please do not change the name of this class
def MyResNet(args):
    return ResNet(ResidualBlock, args=args)

