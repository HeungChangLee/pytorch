# SqueezeNet fitted MNIST dataset by Pytorch
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
            nn.BatchNorm2d(squeeze_dim), #add BatchNorm term after Conv_layer # Because we didn't get good result without BatchNorm term in MNIST dataset
            nn.ReLU())
        self.expand_layer1x1 = nn.Sequential(
            nn.Conv2d(squeeze_dim, expand1x1_dim, kernel_size = 1),
            nn.BatchNorm2d(expand1x1_dim),
            nn.ReLU())
        self.expand_layer3x3 = nn.Sequential(
            nn.Conv2d(squeeze_dim, expand3x3_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(expand1x1_dim),
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
            #nn.Conv2d(3, 96, kernel_size = 7, stride = 2, padding = 1), #conv1 #There isn't "padding = 1" in the paper, but it need to get 111x111 output.
            nn.Conv2d(1, 96, kernel_size = 3, stride = 1, padding = 1), #conv1 #For MNIST images which are 28x28 size.
            nn.BatchNorm2d(96),
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
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(),
            nn.AvgPool2d(3, stride = 1) #For MNIST images which are 28x28 size.
        )
        for m in self.modules():
            weight_init(m)
        
    def forward(self, x):
        output = self.squeezenet(x)
        return (output.view(output.size(0), self.num_classes))


model = SqueezeNet(10)


# MNIST Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

num_epochs = 5
batch_size = 100
learning_rate = 0.001

data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

train_dataset = dsets.MNIST(root='./data/',
                           train=True,
                          #transform=data_transform) #You can use upper data_transform for preprocessing
                           transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root='./data',
                          train=False,
                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)


# Training the SqueezeNet model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


# Evaluate the SqueezeNet model
    model.eval()
    test_loss = 0
    correct = 0
    for images, labels in test_loader:
        images = Variable(images, volatile=True)
        labels = Variable(labels)
        outputs = model(images)
        test_loss += criterion(outputs, labels)
        predict = outputs.data.max(1, keepdim=True)[1]
        correct += predict.eq(labels.data.view_as(predict)).cpu().sum()
        
    test_loss /= len(test_loader.dataset)
    print ('Test Loss: %.4f, Accuracy: %.2f%%' %(test_loss, 100. * correct / len(test_loader.dataset)))