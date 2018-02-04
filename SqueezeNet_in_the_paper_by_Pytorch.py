# SqueezeNet in the paper at ICLR 2017 by Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        #init.xavier_uniform(m.weight.data) # You can use xavier_uniform weight initialization
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0.2)


class Fire_Module(nn.Module):
    def __init__(self, input_dim, squeeze_dim, expand1x1_dim, expand3x3_dim):
        super(Fire_Module, self).__init__()
        self.squeeze_layer = nn.Sequential(
            nn.Conv2d(input_dim, squeeze_dim, kernel_size = 1),
            #How about BatchNorm
            nn.ReLU())
        self.expand_layer1x1 = nn.Sequential(
            nn.Conv2d(squeeze_dim, expand1x1_dim, kernel_size = 1),
            nn.ReLU())
        self.expand_layer3x3 = nn.Sequential(
            nn.Conv2d(squeeze_dim, expand3x3_dim, kernel_size = 3, padding = 1),
            nn.ReLU())
        for m in self.modules():
            weight_init(m)
        
    def forward(self, x):
        output_squeeze = self.squeeze_layer(x)
        output = torch.cat([self.expand_layer1x1(output_squeeze),
                           self.expand_layer3x3(output_squeeze)], dim = 1)
        return (output)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.squeezenet = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 7, stride = 2, padding = 1), #conv1 #There isn't "padding = 1" in the paper, but it need to get 111x111 output.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
            Fire_Module(96, 16, 64, 64), #fire2
            Fire_Module(128, 16, 64, 64), #fire3
            Fire_Module(128, 32, 128, 128), #fire4
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
            Fire_Module(256, 32, 128, 128), #fire5
            Fire_Module(256, 48, 192, 192), #fire6
            Fire_Module(384, 48, 192, 192), #fire7
            Fire_Module(384, 64, 256, 256), #fire8
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
            Fire_Module(512, 64, 256, 256), #fire9
            nn.Dropout(p = 0.5), #after the fire9 module
            nn.Conv2d(512, self.num_classes, kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.AvgPool2d(13, stride = 1)
        )
        for m in self.modules():
            weight_init(m)
        
    def forward(self, x):
        output = self.squeezenet(x)
        return (output.view(output.size(0), self.num_classes))
