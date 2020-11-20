import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init

import sys
sys.path.append('Utils')
from Layers import LRN

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv1.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv1.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.lrn1_1 = nn.Sequential()
        self.lrn1_1.add_module('lrn1_s1_1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        
        self.lrn1_2 = nn.Sequential()
        self.lrn1_2.add_module('lrn1_s1_2',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv2.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv2.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.lrn2_1 = nn.Sequential()
        self.lrn2_1.add_module('lrn2_s1_1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        
        self.lrn2_2 = nn.Sequential()
        self.lrn2_2.add_module('lrn2_s1_2',LRN(local_size=5, alpha=0.0001, beta=0.75))
        
        self.conv = nn.Sequential()
        self.conv.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu4_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu5_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(4096, 4096))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(2*4096, 4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))

        self.regressor = nn.Sequential()
        self.regressor.add_module('fc8',nn.Linear(4096, 4096))
        self.regressor.add_module('relu8',nn.ReLU(inplace=True))
        self.regressor.add_module('drop8',nn.Dropout(p=0.5))
        self.regressor.add_module('fc9',nn.Linear(4096, 3))
        self.regressor.add_module('relu9',nn.ReLU(inplace=True))
        
        #self.apply(weights_init)

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x1, x2):
        B, H, W, C = x1.size()
        x1 = x1.transpose(1,3)
        x2 = x2.transpose(1,3)
        
        z1, z2 = self.conv1(x1.float()), self.conv1(x2.float())
        z1, z2 = self.lrn1_1(z1), self.lrn1_2(z2)
        z1, z2 = self.conv2(z1), self.conv2(z2)
        z1, z2 = self.lrn2_1(z1), self.lrn2_2(z2)
        z1, z2 = self.conv(z1), self.conv(z2)
        z1, z2 = self.fc6(z1.view(B, -1)), self.fc6(z2.view(B, -1))
        
        z_list = [z1,z2]
        #z = z.view([B,1,-1])
        #x_list.append(z)

        x = cat(z_list,1)
        print('Size of cat list', x.size())
        x = self.fc7(x)
        #x = self.fc7(x.view(B,-1))
        x = self.regressor(x)
        print('Each step print', z1.size(), z2.size())
        
        return x


def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)