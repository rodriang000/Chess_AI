# Pos2Vec.py
# initial weights for supervised training
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:00:45 2019

@author: sanbi
"""

from __future__ import print_function
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np

class pos2vec(nn.Module):
    def __init__(self):
        super(pos2vec, self).__init__()
        
        self.fce1 = nn.Linear(773, 100)
        self.bne1 = nn.BatchNorm1d(100)
        self.fce2 = nn.Linear(100, 100)
        self.bne2 = nn.BatchNorm1d(100)
        self.fce3 = nn.Linear(100, 100)
        self.bne3 = nn.BatchNorm1d(100)
        
        self.fcd1 = nn.Linear(100, 100)
        self.bnd1 = nn.BatchNorm1d(100)
        self.fcd2 = nn.Linear(100, 100)
        self.bnd2 = nn.BatchNorm1d(100)
        self.fcd3 = nn.Linear(100, 773)
        self.bnd3 = nn.BatchNorm1d(773)
        
    def encode(self, x):
        x = F.leaky_relu(self.bne1(self.fce1(x)))
        x = F.leaky_relu(self.bne2(self.fce2(x)))
        x = F.leaky_relu(self.bne3(self.fce3(x)))
        return x
    
    def decode(self, z):
        z = F.leaky_relu(self.bnd1(self.fcd1(z)))
        z = F.leaky_relu(self.bnd2(self.fcd2(z)))
        z = F.sigmoid(self.bnd3(self.fcd3(z))) 
        return z
    
    def forward(self, x):
        enc = self.encode(x.view(-1, 773))
        return self.decode(enc), enc
'''    
def loss_function(x_pred, x):
    BCE = F.binary_cross_entropy(x_pred, x.view(-1, 773), size_average=False)
    return BCE

device = torch.device("cpu")

learning_rate = 0.005
decay = 0.98
batch_size = 128

games = np.load('data/bitboards.npy')
np.random.shuffle(games)
train_games = games[:int(len(games)*.8)]
test_games = games[int(len(games)*.8):]

class TrainSet(Dataset):
    def __init__(self):
        pass
    
    def __getitem__(self, index):
        return (torch.from_numpy(train_games[index]).type(torch.FloatTensor), 1)
    
    def __len__(self):
        return train_games.shape[0]
    
class TestSet(Dataset):
    def __init__(self):
        pass
    
    def __getitem__(self, index):
        return (torch.from_numpy(test_games[index]).type(torch.FloatTensor), 1)
    
    def __len__(self):
        return test_games.shape[0]
    
    
train_loader = torch.utils.data.DataLoader(TrainSet(), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestSet(), batch_size=batch_size, shuffle=True)

model = pos2vec().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_epoch = 1
resume = True
'''
'''
for epoch in range(start_epoch, 201):
    #train network
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        data_pred, enc = model(data)
        loss = loss_function(data_pred, data)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('Epoch: {} Average loss: {:4f}'.format(epoch, epoch_loss / len(train_loader.dataset)))
    #save network
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
    save_dir = 'C:\\Users\\sanbi\\Desktop\\AI\\checkpoints\\pos2vec\\lr_{}_decay_{}'.format(int(learning_rate*1000), int(decay*100))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, os.path.join(save_dir, 'pos2vec_{}.pth.tar'.format(epoch)}}))
    #modify learning rate after every epoch
    for params in optimizer.param_groups:
        params['lr'] *= decay
    '''
    
