import torch
import torch.nn.functional as F
import torch.nn as nn

class PhiNet(torch.nn.Module): # [2, 50, 20]
    def __init__(self, n_features, n_hidden, n_output): # 2, 50, 20
        super(PhiNet,self).__init__()
        self.linear1 = nn.Linear(n_features, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_hidden)  #这个也加了一层，可以注释掉

        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)

        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.output(x)
        return x  # [10, 20]

class FNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output): # [1, 50, 1]  n_features指的是x 有1列
        super(FNet, self).__init__()
        self.phinet = PhiNet(2, n_hidden, 20)
        self.linear1 = nn.Linear(n_features + 20, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x, xy):
        b_s = x.shape[0]
        xy = self.phinet(xy) # [10, 20]
        xy = torch.mean(xy, dim=0) # [1, 20]
        xy = xy.expand(b_s, xy.shape[0]) # [10, 20]
        x = torch.cat((x, xy), dim=1)  # [10, 21]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.output(x)
        return x

    

