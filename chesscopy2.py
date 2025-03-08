import numpy as np
import random
from datetime import datetime
import torch
import torch.nn as nn
import chess
import chess.engine
import pickle
import os

random.seed(int(datetime.now().timestamp()))
BOARD_SIZE = 8

POSITIONS = [
    {
        "king": [[0, 4]],
        "queen": [[0, 3]],
        "rook": [[0, 0], [0, 7]],
        "bishop": [[0, 2], [0, 5]],
        "knight": [[0, 1], [0, 6]],
        "pawn": [[1, i] for i in range(BOARD_SIZE)],
    },
    {
        "king": [[7, 3]],
        "queen": [[7, 4]],
        "rook": [[7, 0], [7, 7]],
        "bishop": [[7, 2], [7, 5]],
        "knight": [[7, 1], [7, 6]],
        "pawn": [[6, i] for i in range(BOARD_SIZE)],
    },
]

MAPS = {
    "king": [
        [-3, -4, -4, -5, -5, -4, -4, -3],
        [-3, -4, -4, -5, -5, -4, -4, -3],
        [-3, -4, -4, -5, -5, -4, -4, -3],
        [-3, -4, -4, -5, -5, -4, -4, -3],
        [-2, -3, -3, -4, -4, -3, -3, -2],
        [-1, -2, -2, -3, -3, -2, -2, -1],
        [2, 2, 0, 0, 0, 0, 2, 2],
        [2, 3, 1, 0, 0, 1, 3, 2],
    ],
    "queen": [
        [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1],
        [-0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
        [0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
        [-1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1],
        [-1, 0, 0.5, 0, 0, 0, 0, -1],
        [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
    ],
    "rook": [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 1, 1, 1, 1, 1, 1, 0.5],
        [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
        [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
        [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
        [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
        [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
        [0, 0, 0, 0.5, 0.5, 0, 0, 0],
    ],
    "bishop": [
        [-2, -1, -1, -1, -1, -1, -1, -2],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0.5, 1, 1, 0.5, 0, -1],
        [-1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1],
        [-1, 0, 1, 1, 1, 1, 0, -1],
        [-1, 1, 1, 1, 1, 1, 1, -1],
        [-1, 0.5, 0, 0, 0, 0, 0.5, -1],
        [-2, -1, -1, -1, -1, -1, -1, -2],
    ],
    "knight": [
        [-5, -4, -3, -3, -3, -3, -4, -5],
        [-4, -2, 0, 0, 0, 0, -2, -4],
        [-3, 0, 1, 1.5, 1.5, 1, 0, -3],
        [-3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3],
        [-3, 0, 1.5, 2, 2, 1.5, 0, -3],
        [-3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3],
        [-4, -2, 0, 0.5, 0.5, 0, -2, -4],
        [-5, -4, -3, -3, -3, -3, -4, -5],
    ],
    "pawn": [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 5, 5, 5, 5, 5, 5, 5],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5],
        [0, 0, 0, 2, 2, 0, 0, 0],
        [0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5],
        [0.5, 1, 1, -2, -2, 1, 1, 0.5],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
}

WEIGHTS = {"pawn": 10, "knight": 30, "bishop": 30, "rook": 50, "queen": 90, "king": 900}

# -------------------------------------------------------------------
# Updated CustomChessModel using CNN architecture and new board encoding
# -------------------------------------------------------------------
class CustomChessModel(nn.Module):
    def __init__(self):
        super(CustomChessModel, self).__init__()
        # Input shape: (batch_size, 12, 8, 8)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------------------------------------------------
# Updated board_to_tensor for 12-channel (12,8,8) encoding
# -------------------------------------------------------------------
def board_to_tensor(board):
    """
    Converts a python‑chess board to a tensor of shape (1, 12, 8, 8).
    Channels 0-5 correspond to white pawn, knight, bishop, rook, queen, king,
    and channels 6-11 to the corresponding black pieces.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_to_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,2
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        if piece.color == chess.WHITE:
            channel = piece_to_idx[piece.piece_type]
        else:
            channel = piece_to_idx[piece.piece_type] + 6
        tensor[channel, row, col] = 1.0
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

# -------------------------------------------------------------------
# Updated load_trained_model (remove weights_only and load new architecture)
# -------------------------------------------------------------------
def load_trained_model(model_path=r"C:\Users\Krishna Bansal\OneDrive\Documents\Python\Genetic-Chess-Algorithm-master\Genetic-Chess-Algorithm-master\custom_stockfish_model.pth"):
    model = CustomChessModel()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

trained_model = load_trained_model()

# -------------------------------------------------------------------
# The rest of the code remains largely unchanged: Piece, Player, Board, World, etc.
# -------------------------------------------------------------------
class Piece:
    def __init__(self, value, type_, diag, hv, moveset, x, y, player, map_):
        self.value = value
        self.type_ = type_
        self.diag = diag
        self.hv = hv
        self.moveset = moveset
        self.x = x
        self.y = y
        self.testX = x
        self.testY = y
        self.alive = True
        self.testAlive = True
        self.player = player
        self.map = map_
        self.rawMap = map_

        if player == 0:
            if type_ == "king":
                self.icon = "♚"
            elif type_ == "queen":
                self.icon = "♛"
            elif type_ == "rook":
                self.icon = "♜"
            elif type_ == "bishop":
                self.icon = "♝"
            elif type_ == "knight":
                self.icon = "♞"
            else:
                self.icon = "♟"
        else:
            if type_ == "king":
                self.icon = "♔"
            elif type_ == "queen":
                self.icon = "♕"
            elif type_ == "rook":
                self.icon = "♖"
            elif type_ == "bishop":
                self.icon = "♗"
            elif type_ == "knight":
                self.icon = "♘"
            else:
                self.icon = "♙"

    def getMoves(self, board):
        moves = set([])
        killMoves = set([])
        if self.diag:
            for i in range(1, BOARD_SIZE):
                if self.testY + i < BOARD_SIZE and self.testX + i < BOARD_SIZE:
                    tempPiece = board[self.testY + i][self.testX + i]
                    if tempPiece == False:
                        moves.add((self.testX + i, self.testY + i))
                    elif tempPiece.player != self.player:
                        killMoves.add((self.testX + i, self.testY + i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testY + i < BOARD_SIZE and self.testX - i >= 0:
                    tempPiece = board[self.testY + i][self.testX - i]
                    if tempPiece == False:
                        moves.add((self.testX - i, self.testY + i))
                    elif tempPiece.player != self.player:
                        killMoves.add((self.testX - i, self.testY + i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testY - i >= 0 and self.testX + i < BOARD_SIZE:
                    tempPiece = board[self.testY - i][self.testX + i]
                    if tempPiece == False:
                        moves.add((self.testX + i, self.testY - i))
                    elif tempPiece.player != self.player:
                        killMoves.add((self.testX + i, self.testY - i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testY - i >= 0 and self.testX - i >= 0:
                    tempPiece = board[self.testY - i][self.testX - i]
                    if tempPiece == False:
                        moves.add((self.testX - i, self.testY - i))
                    elif tempPiece.player != self.player:
                        killMoves.add((self.testX - i, self.testY - i))
                        break
                    else:
                        break
                else:
                    break
        if self.hv:
            for i in range(1, BOARD_SIZE):
                if self.testY + i < BOARD_SIZE:
                    tempPiece = board[self.testY + i][self.testX]
                    if tempPiece == False:
                        moves.add((self.testX, self.testY + i))
                    elif tempPiece.player != self.player:
                        killMoves.add((self.testX, self.testY + i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testY - i >= 0:
                    tempPiece = board[self.testY - i][self.testX]
                    if tempPiece == False:
                        moves.add((self.testX, self.testY - i))
                    elif tempPiece.player != self.player:
                        killMoves.add((self.testX, self.testY - i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testX + i < BOARD_SIZE:
                    tempPiece = board[self.testY][self.testX + i]
                    if tempPiece == False:
                        moves.add((self.testX + i, self.testY))
                    elif tempPiece.player != self.player:
                        killMoves.add((self.testX + i, self.testY))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testX - i >= 0:
                    tempPiece = board[self.testY][self.testX - i]
                    if tempPiece == False:
                        moves.add((self.testX - i, self.testY))
                    elif tempPiece.player != self.player:
                        killMoves.add((self.testX - i, self.testY))
                        break
                    else:
                        break
                else:
                    break
        if len(self.moveset) > 0:
            for move in self.moveset:
                if (
                    0 <= self.testX + move[0] < BOARD_SIZE
                    and 0 <= self.testY + move[1] < BOARD_SIZE
                ):
                    if self.type_ == "pawn":
                        tempPiece = board[self.testY + move[1]][self.testX + move[0]]
                        tempPiece1 = False
                        if self.testX + 1 < BOARD_SIZE:
                            tempPiece1 = board[self.testY + move[1]][self.testX + 1]
                        tempPiece2 = False
                        if self.testX - 1 >= 0:
                            tempPiece2 = board[self.testY + move[1]][self.testX - 1]
                        if tempPiece == False:
                            moves.add((self.testX + move[0], self.testY + move[1]))
                        if tempPiece1 and tempPiece1.player != self.player:
                            killMoves.add((self.testX + 1, self.testY + move[1]))
                        if tempPiece2 and tempPiece2.player != self.player:
                            killMoves.add((self.testX - 1, self.testY + move[1]))
                    else:
                        tempPiece = board[self.testY + move[1]][self.testX + move[0]]
                        if tempPiece == False:
                            moves.add((self.testX + move[0], self.testY + move[1]))
                        elif tempPiece.player != self.player:
                            killMoves.add((self.testX + move[0], self.testY + move[1]))
        return [moves, killMoves]

class King(Piece):
    def __init__(self, value, x, y, player, map_):
        Piece.__init__(
            self,
            value,
            "king",
            False,
            False,
            [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]],
            x,
            y,
            player,
            map_,
        )

class Queen(Piece):
    def __init__(self, value, x, y, player, map_):
        Piece.__init__(self, value, "queen", True, True, [], x, y, player, map_)

class Rook(Piece):
    def __init__(self, value, x, y, player, map_):
        Piece.__init__(self, value, "rook", False, True, [], x, y, player, map_)

class Bishop(Piece):
    def __init__(self, value, x, y, player, map_):
        Piece.__init__(self, value, "bishop", True, False, [], x, y, player, map_)

class Knight(Piece):
    def __init__(self, value, x, y, player, map_):
        Piece.__init__(
            self,
            value,
            "knight",
            False,
            False,
            [[-2, -1], [-1, -2], [1, -2], [2, -1], [-2, 1], [-1, 2], [1, 2], [2, 1]],
            x,
            y,
            player,
            map_,
        )

class Pawn(Piece):
    def __init__(self, value, x, y, player, map_):
        Piece.__init__(
            self,
            value,
            "pawn",
            False,
            False,
            [[0, 1 * (-1 * player)]],
            x,
            y,
            player,
            map_,
        )

class Player:
    def __init__(
        self,
        board,
        side,
        pawnVal=10,
        knightVal=30,
        bishopVal=30,
        rookVal=50,
        queenVal=90,
        kingVal=900,
        maps=MAPS,
    ):
        self.board = board
        self.side = side
        self.pieces = []
        self.maps = MAPS
        self.weights = {
            "pawn": pawnVal,
            "knight": knightVal,
            "bishop": bishopVal,
            "rook": rookVal,
            "queen": queenVal,
            "king": kingVal,
        }

        for piece in POSITIONS[side].keys():
            if piece == "king":
                for i in range(len(POSITIONS[side][piece])):
                    self.pieces.append(
                        King(
                            kingVal,
                            POSITIONS[side][piece][i][1],
                            POSITIONS[side][piece][i][0],
                            side,
                            maps[piece],
                        )
                    )
            elif piece == "queen":
                for i in range(len(POSITIONS[side][piece])):
                    self.pieces.append(
                        Queen(
                            queenVal,
                            POSITIONS[side][piece][i][1],
                            POSITIONS[side][piece][i][0],
                            side,
                            maps[piece],
                        )
                    )
            elif piece == "rook":
                for i in range(len(POSITIONS[side][piece])):
                    self.pieces.append(
                        Rook(
                            rookVal,
                            POSITIONS[side][piece][i][1],
                            POSITIONS[side][piece][i][0],
                            side,
                            maps[piece],
                        )
                    )
            elif piece == "bishop":
                for i in range(len(POSITIONS[side][piece])):
                    self.pieces.append(
                        Bishop(
                            bishopVal,
                            POSITIONS[side][piece][i][1],
                            POSITIONS[side][piece][i][0],
                            side,
                            maps[piece],
                        )
                    )
            elif piece == "knight":
                for i in range(len(POSITIONS[side][piece])):
                    self.pieces.append(
                        Knight(
                            knightVal,
                            POSITIONS[side][piece][i][1],
                            POSITIONS[side][piece][i][0],
                            side,
                            maps[piece],
                        )
                    )
            elif piece == "pawn":
                for i in range(len(POSITIONS[side][piece])):
                    self.pieces.append(
                        Pawn(
                            pawnVal,
                            POSITIONS[side][piece][i][1],
                            POSITIONS[side][piece][i][0],
                            side,
                            maps[piece],
                        )
                    )
            else:
                print("ERROR: " + piece)

    def evaluate_board(self, custom_board):
        board_fen = custom_board.printBoard()
        chess_board = chess.Board(board_fen)
        with torch.no_grad():
            tensor_input = board_to_tensor(chess_board)
            return trained_model(tensor_input).item()

    def generateBoard(self):
        board = [[False for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]
        otherPieces = []
        if self.side == 0:
            otherPieces = self.board.p2.pieces
        else:
            otherPieces = self.board.p1.pieces

        for piece in self.pieces:
            if piece.testAlive:
                board[piece.testY][piece.testX] = piece
        for piece in otherPieces:
            if piece.testAlive:
                board[piece.testY][piece.testX] = piece
        return board

    def move(self):
        otherPieces = None
        if self.side == 0:
            otherPieces = self.board.p2.pieces
        else:
            otherPieces = self.board.p1.pieces
        baseVal = self.evaluate_board(self.board)
        maxVal = -100000
        maxMove = (0, 0)
        maxPiece = None
        maxKill = None
        for piece in self.pieces:
            if piece.testAlive:
                board = self.generateBoard()
                moves = piece.getMoves(board)
                newScore = baseVal - piece.map[piece.y][piece.x]
                for killMove in moves[1]:
                    killPiece = board[killMove[1]][killMove[0]]
                    newScore1 = (
                        newScore
                        + killPiece.value
                        + killPiece.map[killMove[1]][killMove[0]]
                        + piece.map[killMove[1]][killMove[0]]
                    )
                    killPiece.testAlive = False
                    piece.testX = killMove[0]
                    piece.testY = killMove[1]
                    minVal = 100000
                    for otherPiece in otherPieces:
                        if otherPiece.testAlive:
                            board1 = self.generateBoard()
                            moves1 = otherPiece.getMoves(board1)
                            newScore2 = (
                                newScore1 + otherPiece.map[otherPiece.y][otherPiece.x]
                            )
                            for killMove1 in moves1[1]:
                                killPiece1 = board1[killMove1[1]][killMove1[0]]
                                newScore3 = (
                                    newScore2
                                    - killPiece1.value
                                    - killPiece1.map[killMove1[1]][killMove1[0]]
                                    - otherPiece.map[killMove1[1]][killMove1[0]]
                                )
                                minVal = min(newScore3, minVal)
                            for move1 in moves1[0]:
                                newScore3 = (
                                    newScore2 - otherPiece.map[move1[1]][move1[0]]
                                )
                                minVal = min(newScore3, minVal)
                    piece.testX = piece.x
                    piece.testY = piece.y
                    killPiece.testAlive = True
                    if minVal > maxVal:
                        maxVal = minVal
                        maxMove = killMove
                        maxPiece = piece
                        maxKill = killPiece
                for move in moves[0]:
                    newScore1 = newScore + piece.map[move[1]][move[0]]
                    piece.testX = move[0]
                    piece.testY = move[1]
                    minVal = 100000
                    for otherPiece in otherPieces:
                        if otherPiece.testAlive:
                            board1 = self.generateBoard()
                            moves1 = otherPiece.getMoves(board1)
                            newScore2 = (
                                newScore1 + otherPiece.map[otherPiece.y][otherPiece.x]
                            )
                            for killMove1 in moves1[1]:
                                killPiece1 = board1[killMove1[1]][killMove1[0]]
                                newScore3 = (
                                    newScore2
                                    - killPiece1.value
                                    - killPiece1.map[killMove1[1]][killMove1[0]]
                                    - otherPiece.map[killMove1[1]][killMove1[0]]
                                )
                                minVal = min(newScore3, minVal)
                            for move1 in moves1[0]:
                                newScore3 = (
                                    newScore2 - otherPiece.map[move1[1]][move1[0]]
                                )
                                minVal = min(newScore3, minVal)
                    piece.testX = piece.x
                    piece.testY = piece.y
                    if minVal > maxVal:
                        maxVal = minVal
                        maxMove = move
                        maxPiece = piece
                        maxKill = None
        if maxKill is not None:
            maxKill.testAlive = False
            maxKill.alive = False
            if maxKill.player == self.side:
                for p in range(len(self.pieces)):
                    if self.pieces[p] is maxKill:
                        del self.pieces[p]
                        break
            else:
                for p in range(len(otherPieces)):
                    if otherPieces[p] is maxKill:
                        del otherPieces[p]
                        break
        maxPiece.map[maxPiece.y][maxPiece.x] -= 3
        maxPiece.x = maxMove[0]
        maxPiece.y = maxMove[1]
        maxPiece.testX = maxMove[0]
        maxPiece.testY = maxMove[1]

class Board:
    def __init__(self, w1=WEIGHTS, w2=WEIGHTS, m1=MAPS, m2=MAPS, printMoves=True):
        self.printMoves = printMoves
        self.board = [[False for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]
        self.p1 = Player(
            self,
            0,
            pawnVal=w1["pawn"],
            knightVal=w1["knight"],
            bishopVal=w1["bishop"],
            rookVal=w1["rook"],
            queenVal=w1["queen"],
            kingVal=w1["king"],
            maps=m1,
        )
        self.p2 = Player(
            self,
            1,
            pawnVal=w2["pawn"],
            knightVal=w2["knight"],
            bishopVal=w2["bishop"],
            rookVal=w2["rook"],
            queenVal=w2["queen"],
            kingVal=w2["king"],
            maps=m2,
        )

    def evaluate_final(self):
        board_fen = self.printBoard()
        try:
            chess_board = chess.Board(board_fen)
        except ValueError:
            return -1000
        with torch.no_grad():
            tensor_input = board_to_tensor(chess_board)
            return trained_model(tensor_input).item()

    def updateBoard(self):
        self.board = [[False for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]
        for piece in self.p1.pieces:
            self.board[piece.y][piece.x] = piece
        for piece in self.p2.pieces:
            self.board[piece.y][piece.x] = piece

    def printBoard(self):
        board = chess.Board()  # Create an empty board
        for piece in self.p1.pieces + self.p2.pieces:
            if piece.alive:
                piece_symbol = (
                    piece.type_[0].lower()
                    if piece.player == 0
                    else piece.type_[0].upper()
                )
                board.set_piece_at(
                    chess.square(piece.x, piece.y),
                    chess.Piece.from_symbol(piece_symbol),
                )
        return board.fen()

    def getWinner(self):
        moveCount = 0
        while moveCount < 1000:
            moveCount += 1
            if not any(piece.type_ == "king" for piece in self.p1.pieces):
                final_eval = self.evaluate_final()
                return ("Stockfish", final_eval, moveCount)
            self.p1.move()
            self.updateBoard()
            if not any(piece.type_ == "king" for piece in self.p2.pieces):
                final_eval = self.evaluate_final()
                return ("GA", final_eval, moveCount)
            self.p2.move()
            self.updateBoard()
        final_eval = self.evaluate_final()
        return ("Stockfish", final_eval, moveCount)

class World:
    def __init__(self, genSize):
        self.size = genSize
        self.generations = {}
        self.genCount = 0

    def mutate(self, board):
        if hasattr(board, "printBoard"):
            new_fen = board.printBoard()
        else:
            new_fen = board.fen()
        new_board = chess.Board(new_fen)
        legal_moves = list(new_board.legal_moves)
        if not legal_moves:
            return new_board
        new_board.push(random.choice(legal_moves))
        return new_board

    def crossover(self, parent1, parent2):
        fen1 = parent1.printBoard().split(" ")[0]
        fen2 = parent2.printBoard().split(" ")[0]
        ranks1 = fen1.split("/")
        ranks2 = fen2.split("/")
        for _ in range(10):
            cp = random.randint(1, BOARD_SIZE - 1)
            new_ranks = ranks1[:cp] + ranks2[cp:]
            new_board_part = "/".join(new_ranks)
            new_fen = new_board_part + " w KQkq - 0 1"
            try:
                new_board = chess.Board(new_fen)
                return new_board
            except Exception as e:
                continue
        return chess.Board(parent1.printBoard())

    def evaluate_board(self, custom_board):
        board_fen = custom_board.printBoard()
        try:
            chess_board = chess.Board(board_fen)
        except ValueError:
            print(f"Invalid FEN encountered:{board_fen}")
            return -1000
        with torch.no_grad():
            tensor_input = board_to_tensor(chess_board)
            return trained_model(tensor_input).item()

    def runGeneration(self):
        print(f"Running Generation {self.genCount + 1}")
        self.generations[self.genCount] = [Board() for _ in range(self.size)]
        match_results = []
        for board in self.generations[self.genCount]:
            result = board.getWinner()
            match_results.append(result)
        if not match_results:
            print("No valid boards to evaluate!")
            return
        for winner, fitness, moves in match_results:
            print(f"Match result: Winner: {winner}, Fitness Score: {fitness}, Moves: {moves}")
        scores = []
        for board in self.generations[self.genCount]:
            fitness = self.evaluate_board(board)
            scores.append((board, fitness))
        if not scores:
            print("No valid boards available for evolution!")
            return
        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [score[0] for score in scores[: self.size // 2]]
        weights = [max(score[1] + 1000, 1) for score in scores[: self.size // 2]]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        new_population = top_candidates[:]
        while len(new_population) < self.size:
            parent1, parent2 = random.choices(top_candidates, weights=weights, k=2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.generations[self.genCount] = new_population
        self.genCount += 1
        print(f"Best Board Eval: {scores[0][1]}")
        print(scores[0][0])

# -------------------------------------------------------------------
# Additional Functions for Legal Move Checking and Saving GA Model
# -------------------------------------------------------------------
def is_move_legal(custom_board, piece, move):
    fen = custom_board.printBoard()
    board = chess.Board(fen)
    start_square = chess.square(piece.x, piece.y)
    end_square = chess.square(move[0], move[1])
    candidate_move = chess.Move(start_square, end_square)
    return candidate_move in board.legal_moves

def save_ga_model(best_board, filename="trained_ga_model.pkl"):
    model_data = {
        "p1_weights": best_board.p1.weights,
        "p1_maps": best_board.p1.maps,
        "p2_weights": best_board.p2.weights,
        "p2_maps": best_board.p2.maps,
    }
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)
    print(f"GA model saved to {filename}")

# -------------------------------------------------------------------
# Functions for Interactive Play Against Bot
# -------------------------------------------------------------------
def compute_personal_trainer_move(chess_board):
    best_move = None
    moves = list(chess_board.legal_moves)
    if not moves:
        return None
    best_eval = -float('inf') if chess_board.turn == chess.WHITE else float('inf')
    for move in moves:
        chess_board.push(move)
        input_tensor = board_to_tensor(chess_board).unsqueeze(0)
        with torch.no_grad():
            eval_score = trained_model(input_tensor).item()
        chess_board.pop()
        if chess_board.turn == chess.WHITE:
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
        else:
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = move
    return best_move

def play_human_vs_bot(mode="personal_trainer"):
    if mode == "stockfish":
        try:
            STOCKFISH_PATH = r"C:\Users\Krishna Bansal\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except Exception as e:
            print("Error initializing Stockfish engine:", e)
            return
    board = chess.Board()
    human_color = input("Choose your color (w/b): ").strip().lower()
    human_is_white = True if human_color == "w" else False
    print("\nStarting game. Enter moves in UCI format (e.g., e2e4).\n")
    while not board.is_game_over():
        print(board)
        if board.turn == (chess.WHITE if human_is_white else chess.BLACK):
            move_uci = input("Your move: ").strip()
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
                    continue
            except Exception as e:
                print("Invalid move format. Try again.")
                continue
        else:
            print("Bot is thinking...")
            if mode == "stockfish":
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
            else:
                bot_move = compute_personal_trainer_move(board)
                if bot_move:
                    board.push(bot_move)
                else:
                    print("Bot could not compute a valid move. Passing turn.")
        print("\n")
    print("Game over!")
    print("Result:", board.result())
    if mode == "stockfish":
        engine.quit()
def main():
    while True:
        print("\nSelect mode:")
        print("1. Run Genetic Algorithm Evolution")
        print("2. Play Interactive Game")
        print("3. Quit")
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            try:
                num_gens = int(input("Enter number of generations to run: ").strip())
            except ValueError:
                print("Invalid input; defaulting to 100 generations.")
                num_gens = 100
            world = World(5)
            for _ in range(num_gens):
                world.runGeneration()
        elif choice == "2":
            print("\nSelect game mode:")
            print("1. Play with Stockfish")
            print("2. Play with Personalized Trainer (GA)")
            sub_choice = input("Enter your choice (1 or 2): ").strip()
            if sub_choice == "1":
                play_human_vs_bot(mode="stockfish")
            elif sub_choice == "2":
                play_human_vs_bot(mode="personal_trainer")
            else:
                print("Invalid selection.")
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
