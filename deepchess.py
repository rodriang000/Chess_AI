# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:31:00 2019

@author: sanbi
"""

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import os
from torch.utils.data import Dataset


class DeepChess(nn.Module):
    def __init__(self):
        super(DeepChess, self).__init__()

        self.fc1 = nn.Linear(200, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 2)
        self.bn3 = nn.BatchNorm1d(2)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return F.sigmoid(x)
'''
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda")

learning_rate = 0.01
decay = 0.99
batch_size = 128

games = np.load('./Features/features.npy')  # path to features extracted
wins = np.load('./Data/labelData7000.npy')  # path to labels

p = np.random.permutation(len(wins))
games = games[p]
wins = wins[p]

train_games = games[:int(len(games) * .8)]
train_wins = wins[:int(len(games) * .8)]
test_games = games[int(len(games) * .8):]
test_wins = wins[int(len(games) * .8):]

train_games_wins = train_games[train_wins == 1]
train_games_losses = train_games[train_wins == -1]

test_games_wins = test_games[test_wins == 1]
test_games_losses = test_games[test_wins == -1]


class TrainSet(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        rand_win = train_games_wins[
            np.random.randint(0, train_games_wins.shape[0])]
        rand_loss = train_games_losses[
            np.random.randint(0, train_games_losses.shape[0])]

        # rand_win = train_games_wins[0]
        # rand_loss = train_games_losses[1234]

        order = np.random.randint(0, 2)
        if order == 0:
            stacked = np.hstack((rand_win, rand_loss))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
            return (stacked, label)
        else:
            stacked = np.hstack((rand_loss, rand_win))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
            return (stacked, label)

    def __len__(self):
        return self.length


class TestSet(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        rand_win = test_games_wins[np.random.randint(0, test_games_wins.shape[0])]
        rand_loss = test_games_losses[np.random.randint(0, test_games_losses.shape[0])]

        order = np.random.randint(0, 2)
        if order == 0:
            stacked = np.hstack((rand_win, rand_loss))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
            return (stacked, label)

    def __len__(self):
        return self.length

train_loader = torch.utils.data.DataLoader(TrainSet(100), batch_size=batch_size, shuffle=True)  # input length, what should it be?
test_loader = torch.utils.data.DataLoader(TestSet(100), batch_size=batch_size, shuffle=True)

model = DeepChess().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def loss_function(pred, label):
    return F.binary_cross_entropy(pred, label, size_average=False)
    '''
'''
start_epoch = 1
resume = False  # change to true if continuing training
if resume:
    state = torch.load('./Model/lr_5_decay_98/pos2vec_315.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']

for epoch in range(start_epoch, 1001):
    # train network
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        # Error here? data shape is 100 * 200
        label = label.to(device)

        optimizer.zero_grad()

        pred = model(data)
        loss = loss_function(pred, label)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    # test network at every epoch
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data) # This line is giving errors
            test_loss += loss_function(pred, label).item()
        print('Test set loss: {:.4f}'.format(test_loss / len(test_loader.dataset)))

    # save network at every epoch
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
    save_dir = './Data/lr_{}_decay_{}'.format(int(learning_rate * 100), int(decay * 100))  # path to dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, os.path.join(save_dir, 'deepchess_{}.pth.tar'.format(epoch)))

    # adjust learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        '''
