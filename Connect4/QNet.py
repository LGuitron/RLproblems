import torch
import torch.nn.functional as F

class QNet(torch.nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        self.conv_bn = torch.nn.BatchNorm2d(32)                          #Normalization for 8 filter convolution layers
        self.conv_final = torch.nn.BatchNorm2d(1)                       #Normalization for final convolution layer

        self.initial_conv = torch.nn.Conv2d(2, 32, 3, padding=1)         #32 3x3 filters
        self.res_conv = torch.nn.Conv2d(32, 32, 3, padding=1)             #32 3x3 filters
        self.final_conv = torch.nn.Conv2d(32,1,1)                        #Final convolutional layer (1 1x1 filter)

        self.fc1 = torch.nn.Linear(42, 64)                             #Connect all 336 units to 64 hidden unitss
        self.fc2 = torch.nn.Linear(64, 7)                               #Connect all 64 hidden units to 7 possible outputs

    def forward(self, x):
        x = F.relu(self.conv_bn(self.initial_conv(x)))                  #Initial convolution

        for i in range(6):                                              #Amount of residual blocks to be applied
            temp = F.relu(self.conv_bn(self.res_conv(x)))
            temp = self.conv_bn(self.res_conv(temp))
            x = F.relu(temp+x)

        x = F.relu(self.conv_final(self.final_conv(x)))
        x = x.view(-1, 42)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''
class QNet(torch.nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        self.conv_bn = torch.nn.BatchNorm2d(8)                          #Normalization for 8 filter convolution layers
        self.conv_final = torch.nn.BatchNorm2d(1)                       #Normalization for final convolution layer

        self.initial_conv = torch.nn.Conv2d(2, 8, 3, padding=1)         #32 3x3 filters
        self.res_conv = torch.nn.Conv2d(8, 8, 3, padding=1)             #32 3x3 filters
        self.final_conv = torch.nn.Conv2d(8,1,1)                        #Final convolutional layer (1 1x1 filter)

        self.fc1 = torch.nn.Linear(42, 32)                             #Connect all 336 units to 64 hidden unitss
        self.fc2 = torch.nn.Linear(32, 7)                               #Connect all 64 hidden units to 7 possible outputs

    def forward(self, x):
        x = F.relu(self.conv_bn(self.initial_conv(x)))                  #Initial convolution

        for i in range(5):                                              #Amount of residual blocks to be applied
            temp = F.relu(self.conv_bn(self.res_conv(x)))
            temp = self.conv_bn(self.res_conv(temp))
            x = F.relu(temp+x)

        x = F.relu(self.conv_final(self.final_conv(x)))
        x = x.view(-1, 42)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''
