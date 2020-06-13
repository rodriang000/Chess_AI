# project.py
# implements chess with human player and ai
import chess
import chess.svg
import re
import os
import numpy as np

from convertBoard import get_bitboard
from Pos2Vec import pos2vec
from deepchess import DeepChess
from featurize import featurize


device = torch.device("cpu")
featurizer = pos2vec().to(device)
comparator = DeepChess().to(device)

pos2vec_state_dict = torch.load('', map_location=lambda storage, loc: storage)
deepChess_state_dict = torch.load('', map_location=lambda storage, loc: storage)

featurizer.load_state_dict(pos2vec_state_dict['state_dict'])
comparator.load_state_dict(deepChess_state_dict['state_dict'])

def compare(features):
    features = torch.from_numpy(features).type(torch.FloatTensor)
    return comparator(features).detach().numpy()


def player(board):
    while True:
        moveFrom = input("P1:What piece do you want to move\n")
        if re.search('[a-h]+', moveFrom) and re.search('[1-8]+',moveFrom):
            break
    while True:
        moveTo = input("Where do you want to move your piece\n")
        if re.search('[a-h]+', moveFrom) and re.search('[1-8]+',moveTo):
            break
    move = chess.Move.from_uci(moveFrom + moveTo)
    if move in board.legal_moves:
        board.push(move)
    return board


def ai(board):
    # generate and store possible moves
    moves = board.generate_legal_moves()
    moves = list(moves)

    # store bitboards
    bitboards = []

    for m in moves:
        curr_board = board.copy()
        curr_board.push(m)
        # add bitboard
        bitboards.append(get_bitboard(curr_board))

    bitboards = np.array(bitboards)
    curr_bitboard = get_bitboard(board)

    # obtain features from bitboards
    _, fts = featurize(bitboards)
    fts = fts.detach().numpy()

    # obtain features from board
    _, curr_fts = featurize(curr_bitboard)
    curr_fts = curr_fts.detach().numpy()

    # compare boards
    c = np.hstack((np.repeat(curr_fts, len(moves), axis=0), fts))

    # get scores
    scores = compare(c)
    scores = scores[:, 0]

    best = np.argmax(scores)
    board.push(moves[best])

    return board


def game():
    determinePlayer = 0;
    board = chess.Board()
    # check game state
    while board.is_game_over() == False: 
        print(board)

        if determinePlayer % 2 == 1:
            board = player(board)
        else:
            board = ai(board)
        determinePlayer = determinePlayer + 1

    print(board)
    print("Game Over")


game()
