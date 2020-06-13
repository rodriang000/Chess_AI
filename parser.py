# Angel Rodriguez
# pgn parser to sift through pgn data and save board states.
# 11/24/2019
import chess.pgn
from convertBoard import get_bitboard
import numpy as np
import sys
# Global variables
game_amount = 275606

# Import grandmaster games
pgn = open("GMallboth/GMallboth.pgn")
total_games = []

# Read the games
for i in range(0, 100000):
    print("Working on game: " + str(i))
    game = chess.pgn.read_game(pgn)
    # Save games that don't end in a draw
    if game.headers["Result"] != "1/2-1/2":
        total_games.append(game)

# Try to go through games, save board positions
def saveGamePositions(total_games):
    bit_boards = []
    labels = []
    game_count = 0
    for j in range(0, len(total_games)):
        played_game = total_games[j]
        possible_moves = []
        result = played_game.headers['Result'].split('-')
        if result[0] == '1':
            result = 1
        elif result[0] == '0':
            result = -1
        else:
            result = 0
        board = played_game.board()
        counter = 0
        # Go through each game, saving positions
        for move in played_game.mainline_moves():
            if counter <= 5:
                # Skip the first 5 moves
                board.push(move)
            else:
                # Not a capture move? Save to file
                if not board.is_capture(move):
                    labels.append(result)
                    bit_board = get_bitboard(board)
                    bit_boards.append(bit_board)
                board.push(move)
            counter += 1
        print(game_count)
        if game_count == 6999: # Stop after 7000 games.
            print("Saving game: " + str(j))
            bit_boards = np.array(bit_boards)
            labels = np.array(labels)
            np.save("./Data/gameData.npy", bit_boards)
            np.save("./Data/labelData.npy", labels)
            sys.exit()
        game_count += 1

saveGamePositions(total_games)

# Bitboard representation reference:
# https://github.com/dangeng/DeepChess/blob/master/parse_data.py