import numpy as np
import random
from datetime import datetime
import torch
import torch.nn as nn
import chess
import random
import numpy as np


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


# Load the trained model
class CustomChessModel(nn.Module):
    def __init__(self):
        super(CustomChessModel, self).__init__()
        self.fc1 = nn.Linear(64, 256)  # Input size is 64 (chess board encoding)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)  # Evaluation score output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)
        return x


# --- Then define helper functions ---
def board_to_tensor(board):
    tensor = np.zeros((64,))
    for i, piece in board.piece_map().items():
        tensor[i] = (
            piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
        )
    return torch.tensor(tensor, dtype=torch.float32)


def load_trained_model(
    model_path=r"C:\Users\Krishna Bansal\OneDrive\Documents\Python\custom_stockfish_model.pth",
):
    model = CustomChessModel()  # No base_model argument needed now
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

    # Load trained neural network model


trained_model = load_trained_model()


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
        board_fen = custom_board.printBoard()  # Ensure this function returns FEN
        chess_board = chess.Board(board_fen)  # Convert to chess.Board()
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

        return board.fen()  # Return valid FEN

    def getWinner(self):
        moveCount = 0
        while moveCount < 1000:
            moveCount += 1
            if not any(piece.type_ == "king" for piece in self.p1.pieces):
                final_eval = (
                    self.evaluate_final()
                )  # Define this helper to evaluate final board state
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
        # If the board has printBoard(), use it; otherwise, use board.fen()
        if hasattr(board, "printBoard"):
            new_fen = board.printBoard()
        else:
            new_fen = board.fen()
        new_board = chess.Board(new_fen)  # Create new board from FEN
        legal_moves = list(new_board.legal_moves)
        if not legal_moves:
            return new_board
        new_board.push(random.choice(legal_moves))
        return new_board

    def crossover(self, parent1, parent2):
        # Get FEN strings from both parents (only the board part)
        fen1 = parent1.printBoard().split(" ")[
            0
        ]  # e.g. "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        fen2 = parent2.printBoard().split(" ")[0]

        ranks1 = fen1.split("/")
        ranks2 = fen2.split("/")

        # Try a few random crossover points until we generate a valid FEN.
        for _ in range(10):
            # Choose a random crossover point between 1 and 7 (8 ranks total)
            cp = random.randint(1, BOARD_SIZE - 1)
            new_ranks = ranks1[:cp] + ranks2[cp:]
            new_board_part = "/".join(new_ranks)
            # Use default active color, castling, en passant etc.
            new_fen = new_board_part + " w KQkq - 0 1"
            try:
                new_board = chess.Board(new_fen)
                return new_board
            except Exception as e:
                continue

        # If none of the attempts yielded a valid board, fallback to one parent's board.
        return chess.Board(parent1.printBoard())

    def evaluate_board(self, custom_board):
        board_fen = custom_board.printBoard()  # Get FEN string
        try:
            chess_board = chess.Board(board_fen)
        except ValueError:
            print(f"Invalid FEN encountered:{board_fen}")
            return -1000
        # Convert to chess.Board()
        with torch.no_grad():
            tensor_input = board_to_tensor(chess_board)  # Convert to tensor
            return trained_model(tensor_input).item()  # Get NN score

    def runGeneration(self):
        print(f"Running Generation {self.genCount + 1}")

        # Generate random positions (initial population)
        self.generations[self.genCount] = [Board() for _ in range(self.size)]

        # Simulate games on each board and collect match results:
        match_results = []
        for board in self.generations[self.genCount]:
            # getWinner() should return a tuple: (winner, fitness_score, move_count)
            result = board.getWinner()
            match_results.append(result)

        if not match_results:
            print("No valid boards to evaluate!")
            return

        # Print match results for each board
        for winner, fitness, moves in match_results:
            print(
                f"Match result: Winner: {winner}, Fitness Score: {fitness}, Moves: {moves}"
            )

        # For GA evolution, we can recalculate fitness scores (or use the ones above)
        # Here we recalc using our evaluation function
        scores = []
        for board in self.generations[self.genCount]:
            # Use evaluate_board() to get a fitness score for selection
            fitness = self.evaluate_board(board)
            scores.append((board, fitness))

        # If scores are empty, skip evolution
        if not scores:
            print("No valid boards available for evolution!")
            return

        scores.sort(key=lambda x: x[1], reverse=True)

        # Select top candidates (we use the top half)
        top_candidates = [score[0] for score in scores[: self.size // 2]]

        # Use weighted selection based on fitness scores (shifted to avoid negatives)
        weights = [max(score[1] + 1000, 1) for score in scores[: self.size // 2]]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Create a new population via crossover & mutation from top candidates
        new_population = top_candidates[:]
        while len(new_population) < self.size:
            parent1, parent2 = random.choices(top_candidates, weights=weights, k=2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.generations[self.genCount] = new_population  # Update the population
        self.genCount += 1  # Increment generation count

        print(f"Best Board Eval: {scores[0][1]}")
        print(scores[0][0])


if __name__ == "__main__":
    world = World(5)
    for _ in range(100):
        world.runGeneration()
