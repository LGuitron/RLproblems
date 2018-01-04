import torch
import torch.nn.functional as F

class QNet(torch.nn.Module):
    
    def __init__(self):
        super(QNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 64, 3)                      #64 3x3 filters (4x5)  
        self.conv2 = torch.nn.Conv2d(64, 128, 2)                    #128 3x3 filters (3x4) 
        self.conv3 = torch.nn.Conv2d(128, 256, 2)                   #256 3x3 filters (2x3) 
        self.conv4 = torch.nn.Conv2d(256, 512, 2)                   #512 3x3 filters (1x2) 
        self.fc = torch.nn.Linear(1024, 7)                          #Connect all 512 hidden units to 7 possible outputs
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 1024)
        x = F.relu(self.fc(x))
        
        return x
