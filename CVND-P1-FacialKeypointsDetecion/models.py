import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5) # output = (32, 220, 220)     
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2)  # output = (32, 110, 110)
        self.conv1_drop = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(32, 64, 5) # output = (64, 106, 106) -> # mp output = (64, 53, 53)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout(p=0.3)
        
        self.conv3 = nn.Conv2d(64, 128, 5) # output = (128, 49, 49) -> # mp output = (128, 24, 24)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_drop = nn.Dropout(p=0.3)


        self.conv4 = nn.Conv2d(128, 256, 5) # output = (256, 20, 20) -> # mp output = (256, 10, 10)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_drop = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(256*10*10,2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc1_drop = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.fc2_drop = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(1024, 136)
        
    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.conv1_drop(x)
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.conv2_drop(x)
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.conv3_drop(x)
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.conv4_drop(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.fc1_drop(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        
        return x
