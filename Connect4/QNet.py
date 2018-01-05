import torch
import torch.nn.functional as F

class QNet(torch.nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        self.conv_bn = torch.nn.BatchNorm2d(32)
        self.initial_conv = torch.nn.Conv2d(2, 32, 3, padding=1)        #32 3x3 filters
        self.res_conv = torch.nn.Conv2d(32, 32, 3, padding=1)           #32 3x3 filters (4x5)
        self.fc = torch.nn.Linear(1344, 7)                              #Connect all 1344 hidden units to 7 possible outputs

    def forward(self, x):
        x = F.relu(self.conv_bn(self.initial_conv(x)))                  #Initial convolution

        for i in range(5):                                              #Amount of residual blocks to be applied
            temp = F.relu(self.conv_bn(self.res_conv(x)))
            temp = self.conv_bn(self.res_conv(temp))
            x = F.relu(temp+x)

        x = x.view(-1, 1344)
        x = F.relu(self.fc(x))
        return x
