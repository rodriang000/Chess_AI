# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:23:06 2019

@author: sanbi
"""

from Pos2Vec import pos2vec
import numpy as np
import torch

model = pos2vec()
state = torch.load('./Model/lr_5_decay_98/pos2vec_315.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])
games = np.load('./Data/gameData7000.npy') #change this to bitboards.npy
# print(len(games))
#48 divides no. of samples evenly (idk why this is important)
batched_games = np.split(games, 9)
#idk if batched_games is even necessary

def featurize(game):
    pred, enc = model(torch.from_numpy(game).type(torch.FloatTensor))
    return enc.detach().numpy()


feat_games = [featurize(batch) for batch in batched_games] #change games to batched_games if that makes a difference
featurized = np.vstack(feat_games)

np.save('./Features/features.npy', featurized) #change file to the path to save to