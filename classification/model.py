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
    def __init__(self, ResidualBlock, num_classes = 24):
        super(ResNet, self).__init__()
        self.inchannel = 64 #
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size = 3, stride = 1, #
                                            padding = 1, bias = False), 
                                  nn.BatchNorm2d(64), #
                                  nn.ReLU())
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride = 2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride = 2)
        # self.layer5 = self.make_layer(ResidualBlock, 1024, 2, stride = 2)
        # self.layer6 = self.make_layer(ResidualBlock, 2048, 2, stride = 2)
        self.maxpool = nn.MaxPool2d((1,4))
        self.fc = nn.Linear(3584*4, num_classes)
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels #match output channel to next input channel 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        #x = self.layer6(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# please do not change the name of this class
def MyResNet():
    return ResNet(ResidualBlock)

