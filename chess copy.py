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


class CustomChessModel(nn.Module):
    def __init__(self):
        super(CustomChessModel, self).__init__()
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
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_to_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        channel = (
            piece_to_idx[piece.piece_type]
            if piece.color == chess.WHITE
            else piece_to_idx[piece.piece_type] + 6
        )
        tensor[channel, row, col] = 1.0
    return torch.tensor(tensor, dtype=torch.float32)


def load_trained_model(
    model_path=r"C:\Users\Krishna Bansal\OneDrive\Documents\Python\Genetic-Chess-Algorithm-master\Genetic-Chess-Algorithm-master\custom_stockfish_model.pth",
):
    model = CustomChessModel()
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Return an untrained model as fallback
        return model


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

    def getMoves(self, board_matrix):
        moves = set([])
        killMoves = set([])
        if self.diag:
            for i in range(1, BOARD_SIZE):
                if self.testY + i < BOARD_SIZE and self.testX + i < BOARD_SIZE:
                    temp = board_matrix[self.testY + i][self.testX + i]
                    if temp is False:
                        moves.add((self.testX + i, self.testY + i))
                    elif temp.player != self.player:
                        killMoves.add((self.testX + i, self.testY + i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testY + i < BOARD_SIZE and self.testX - i >= 0:
                    temp = board_matrix[self.testY + i][self.testX - i]
                    if temp is False:
                        moves.add((self.testX - i, self.testY + i))
                    elif temp.player != self.player:
                        killMoves.add((self.testX - i, self.testY + i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testY - i >= 0 and self.testX + i < BOARD_SIZE:
                    temp = board_matrix[self.testY - i][self.testX + i]
                    if temp is False:
                        moves.add((self.testX + i, self.testY - i))
                    elif temp.player != self.player:
                        killMoves.add((self.testX + i, self.testY - i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testY - i >= 0 and self.testX - i >= 0:
                    temp = board_matrix[self.testY - i][self.testX - i]
                    if temp is False:
                        moves.add((self.testX - i, self.testY - i))
                    elif temp.player != self.player:
                        killMoves.add((self.testX - i, self.testY - i))
                        break
                    else:
                        break
                else:
                    break
        if self.hv:
            for i in range(1, BOARD_SIZE):
                if self.testY + i < BOARD_SIZE:
                    temp = board_matrix[self.testY + i][self.testX]
                    if temp is False:
                        moves.add((self.testX, self.testY + i))
                    elif temp.player != self.player:
                        killMoves.add((self.testX, self.testY + i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testY - i >= 0:
                    temp = board_matrix[self.testY - i][self.testX]
                    if temp is False:
                        moves.add((self.testX, self.testY - i))
                    elif temp.player != self.player:
                        killMoves.add((self.testX, self.testY - i))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testX + i < BOARD_SIZE:
                    temp = board_matrix[self.testY][self.testX + i]
                    if temp is False:
                        moves.add((self.testX + i, self.testY))
                    elif temp.player != self.player:
                        killMoves.add((self.testX + i, self.testY))
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, BOARD_SIZE):
                if self.testX - i >= 0:
                    temp = board_matrix[self.testY][self.testX - i]
                    if temp is False:
                        moves.add((self.testX - i, self.testY))
                    elif temp.player != self.player:
                        killMoves.add((self.testX - i, self.testY))
                        break
                    else:
                        break
                else:
                    break
        if len(self.moveset) > 0:
            for move in self.moveset:
                newX = self.testX + move[0]
                newY = self.testY + move[1]
                if 0 <= newX < BOARD_SIZE and 0 <= newY < BOARD_SIZE:
                    if self.type_ == "pawn":
                        temp = board_matrix[newY][newX]
                        temp1 = (
                            board_matrix[newY][self.testX + 1]
                            if self.testX + 1 < BOARD_SIZE
                            else False
                        )
                        temp2 = (
                            board_matrix[newY][self.testX - 1]
                            if self.testX - 1 >= 0
                            else False
                        )
                        if temp is False:
                            moves.add((newX, newY))
                        if temp1 and temp1.player != self.player:
                            killMoves.add((self.testX + 1, newY))
                        if temp2 and temp2.player != self.player:
                            killMoves.add((self.testX - 1, newY))
                    else:
                        temp = board_matrix[newY][newX]
                        if temp is False:
                            moves.add((newX, newY))
                        elif temp.player != self.player:
                            killMoves.add((newX, newY))
        return [moves, killMoves]


class King(Piece):
    def __init__(self, value, x, y, player, map_):
        super().__init__(
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
        super().__init__(value, "queen", True, True, [], x, y, player, map_)


class Rook(Piece):
    def __init__(self, value, x, y, player, map_):
        super().__init__(value, "rook", False, True, [], x, y, player, map_)


class Bishop(Piece):
    def __init__(self, value, x, y, player, map_):
        super().__init__(value, "bishop", True, False, [], x, y, player, map_)


class Knight(Piece):
    def __init__(self, value, x, y, player, map_):
        super().__init__(
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
        super().__init__(
            value, "pawn", False, False, [[0, 1 * (-1 * player)]], x, y, player, map_
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
        self.maps = maps
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
                for pos in POSITIONS[side][piece]:
                    self.pieces.append(King(kingVal, pos[1], pos[0], side, maps[piece]))
            elif piece == "queen":
                for pos in POSITIONS[side][piece]:
                    self.pieces.append(
                        Queen(queenVal, pos[1], pos[0], side, maps[piece])
                    )
            elif piece == "rook":
                for pos in POSITIONS[side][piece]:
                    self.pieces.append(Rook(rookVal, pos[1], pos[0], side, maps[piece]))
            elif piece == "bishop":
                for pos in POSITIONS[side][piece]:
                    self.pieces.append(
                        Bishop(bishopVal, pos[1], pos[0], side, maps[piece])
                    )
            elif piece == "knight":
                for pos in POSITIONS[side][piece]:
                    self.pieces.append(
                        Knight(knightVal, pos[1], pos[0], side, maps[piece])
                    )
            elif piece == "pawn":
                for pos in POSITIONS[side][piece]:
                    self.pieces.append(Pawn(pawnVal, pos[1], pos[0], side, maps[piece]))

    def evaluate_board(self, custom_board):
        board_fen = custom_board.printBoard()
        chess_board = chess.Board(board_fen)
        with torch.no_grad():
            tensor_input = board_to_tensor(chess_board).unsqueeze(0)
            return trained_model(tensor_input).item()

    def generateBoard(self):
        matrix = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        other = self.board.p2.pieces if self.side == 0 else self.board.p1.pieces
        for p in self.pieces:
            if p.testAlive:
                matrix[p.testY][p.testX] = p
        for p in other:
            if p.testAlive:
                matrix[p.testY][p.testX] = p
        return matrix

    def move(self):
        other = self.board.p2.pieces if self.side == 0 else self.board.p1.pieces
        baseVal = self.evaluate_board(self.board)
        maxVal = -100000
        maxMove = (0, 0)
        maxPiece = None
        maxKill = None
        for p in self.pieces:
            if p.testAlive:
                mat = self.generateBoard()
                moves, killMoves = p.getMoves(mat)
                newScore = baseVal - p.map[p.y][p.x]
                for km in killMoves:
                    kp = mat[km[1]][km[0]]
                    newScore1 = (
                        newScore + kp.value + kp.map[km[1]][km[0]] + p.map[km[1]][km[0]]
                    )
                    kp.testAlive = False
                    p.testX, p.testY = km[0], km[1]
                    minVal = 100000
                    for op in other:
                        if op.testAlive:
                            mat1 = self.generateBoard()
                            moves1, killMoves1 = op.getMoves(mat1)
                            newScore2 = newScore1 + op.map[op.y][op.x]
                            for km1 in killMoves1:
                                kp1 = mat1[km1[1]][km1[0]]
                                newScore3 = (
                                    newScore2
                                    - kp1.value
                                    - kp1.map[km1[1]][km1[0]]
                                    - op.map[km1[1]][km1[0]]
                                )
                                minVal = min(newScore3, minVal)
                            for m1 in moves1:
                                newScore3 = newScore2 - op.map[m1[1]][m1[0]]
                                minVal = min(newScore3, minVal)
                    p.testX, p.testY = p.x, p.y
                    kp.testAlive = True
                    if minVal > maxVal:
                        maxVal = minVal
                        maxMove = km
                        maxPiece = p
                        maxKill = kp
                for m in moves:
                    newScore1 = newScore + p.map[m[1]][m[0]]
                    p.testX, p.testY = m[0], m[1]
                    minVal = 100000
                    for op in other:
                        if op.testAlive:
                            mat1 = self.generateBoard()
                            moves1, killMoves1 = op.getMoves(mat1)
                            newScore2 = newScore1 + op.map[op.y][op.x]
                            for km1 in killMoves1:
                                kp1 = mat1[km1[1]][km1[0]]
                                newScore3 = (
                                    newScore2
                                    - kp1.value
                                    - kp1.map[km1[1]][km1[0]]
                                    - op.map[km1[1]][km1[0]]
                                )
                                minVal = min(newScore3, minVal)
                            for m1 in moves1:
                                newScore3 = newScore2 - op.map[m1[1]][m1[0]]
                                minVal = min(newScore3, minVal)
                    p.testX, p.testY = p.x, p.y
                    if minVal > maxVal:
                        maxVal = minVal
                        maxMove = m
                        maxPiece = p
                        maxKill = None
        if maxKill is not None:
            maxKill.testAlive = False
            maxKill.alive = False
            if maxKill.player == self.side:
                for i, pp in enumerate(self.pieces):
                    if pp is maxKill:
                        del self.pieces[i]
                        break
            else:
                for i, pp in enumerate(other):
                    if pp is maxKill:
                        del other[i]
                        break
        maxPiece.map[maxPiece.y][maxPiece.x] -= 3
        maxPiece.x, maxPiece.y = maxMove[0], maxMove[1]
        maxPiece.testX, maxPiece.testY = maxMove[0], maxMove[1]


class Board:
    def __init__(self, w1=WEIGHTS, w2=WEIGHTS, m1=MAPS, m2=MAPS, printMoves=True):
        self.printMoves = printMoves
        self.board = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
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
        # (Optionally, call self.updateBoard() here if needed)

        def printBoard(self):
            """Returns a FEN string of the current position."""
            b = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")  # Empty board
            b.clear_board()  # Clear to be safe

            for piece in self.p1.pieces + self.p2.pieces:
                if piece.alive:
                    piece_type = None
                    if piece.type_ == "pawn":
                        piece_type = chess.PAWN
                    elif piece.type_ == "knight":
                        piece_type = chess.KNIGHT
                    elif piece.type_ == "bishop":
                        piece_type = chess.BISHOP
                    elif piece.type_ == "rook":
                        piece_type = chess.ROOK
                    elif piece.type_ == "queen":
                        piece_type = chess.QUEEN
                    elif piece.type_ == "king":
                        piece_type = chess.KING

                    color = chess.WHITE if piece.player == 0 else chess.BLACK
                    square = chess.square(
                        piece.x, 7 - piece.y
                    )  # Convert to chess.Square
                    b.set_piece_at(square, chess.Piece(piece_type, color))

            return b.fen()

    def printBoard(self):
        """Returns a FEN string of the current position."""
        b = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")  # Empty board
        b.clear_board()  # Clear to be safe

        for piece in self.p1.pieces + self.p2.pieces:
            if piece.alive:
                piece_type = None
                if piece.type_ == "pawn":
                    piece_type = chess.PAWN
                elif piece.type_ == "knight":
                    piece_type = chess.KNIGHT
                elif piece.type_ == "bishop":
                    piece_type = chess.BISHOP
                elif piece.type_ == "rook":
                    piece_type = chess.ROOK
                elif piece.type_ == "queen":
                    piece_type = chess.QUEEN
                elif piece.type_ == "king":
                    piece_type = chess.KING

                color = chess.WHITE if piece.player == 0 else chess.BLACK
                square = chess.square(piece.x, 7 - piece.y)  # Convert to chess.Square
                b.set_piece_at(square, chess.Piece(piece_type, color))

        return b.fen()

    def updateBoard(self):
        """Updates the internal 2D array representation if you need it."""
        self.board = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for p in self.p1.pieces:
            if p.alive:
                self.board[p.y][p.x] = p
        for p in self.p2.pieces:
            if p.alive:
                self.board[p.y][p.x] = p

    def evaluate_final(self):
        """Evaluates the board using the neural model, if desired."""
        fen = self.printBoard()
        cb = chess.Board(fen)
        with torch.no_grad():
            t = board_to_tensor(cb).unsqueeze(0)
            return trained_model(t).item()

    def getWinner(self):
        """
        Simulates moves for both players until one king is missing or
        until a maximum number of moves is reached.
        Returns a tuple: (winner_name, evaluation_score, move_count).
        """
        moveCount = 0
        while moveCount < 1000:
            moveCount += 1
            # Check if Player 1 (p1) has lost its king
            if not any(p.type_ == "king" for p in self.p1.pieces):
                return ("Stockfish", self.evaluate_final(), moveCount)
            # p1 moves
            self.p1.move()
            self.updateBoard()

            # Check if Player 2 (p2) has lost its king
            if not any(p.type_ == "king" for p in self.p2.pieces):
                return ("GA", self.evaluate_final(), moveCount)
            # p2 moves
            self.p2.move()
            self.updateBoard()

        # If we exit the loop, we default to a draw or some fallback:
        return ("Stockfish", self.evaluate_final(), moveCount)


class World:
    def __init__(self, genSize, initial_population=None, initial_gen_count=0):
        self.size = genSize
        self.generations = {}
        self.genCount = initial_gen_count
        if initial_population is not None:
            self.generations[self.genCount] = initial_population

    def mutate(self, board):
        new_fen = board.printBoard() if hasattr(board, "printBoard") else board.fen()
        nb = chess.Board(new_fen)
        legal = list(nb.legal_moves)
        if not legal:
            return Board() if not hasattr(board, "printBoard") else board
        nb.push(random.choice(legal))
        # Return a custom Board object
        new_board = Board()
        return new_board

    def crossover(self, parent1, parent2):
        fen1 = parent1.printBoard().split(" ")[0]
        fen2 = parent2.printBoard().split(" ")[0]
        r1 = fen1.split("/")
        r2 = fen2.split("/")
        for _ in range(10):
            cp = random.randint(1, BOARD_SIZE - 1)
            new_r = r1[:cp] + r2[cp:]
            new_fen = "/".join(new_r) + " w KQkq - 0 1"
            try:
                # Create a chess.Board to validate the FEN
                chess_board = chess.Board(new_fen)
                # But return a custom Board object
                return Board()  # Use parent's weights and maps if needed
            except Exception:
                continue
        return Board()  # Fallback to a new board

    def evaluate_board(self, custom_board):
        fen = custom_board.printBoard()
        try:
            cb = chess.Board(fen)
        except ValueError:
            print("Invalid FEN:", fen)
            return -1000
        with torch.no_grad():
            t = board_to_tensor(cb).unsqueeze(0)
            return trained_model(t).item()

    def runGeneration(self):
        print(f"Running Generation {self.genCount + 1}")
        if self.genCount not in self.generations:
            self.generations[self.genCount] = [Board() for _ in range(self.size)]
        results = []
        for b in self.generations[self.genCount]:
            results.append(b.getWinner())
        if not results:
            print("No valid boards!")
            return None
        for w, f, m in results:
            print(f"Winner: {w}, Fitness: {f}, Moves: {m}")
        scores = []
        for b in self.generations[self.genCount]:
            scores.append((b, self.evaluate_board(b)))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [s[0] for s in scores[: self.size // 2]]
        weights = [max(s[1] + 1000, 1) for s in scores[: self.size // 2]]
        total = sum(weights)
        weights = [w / total for w in weights]
        new_pop = top[:]
        while len(new_pop) < self.size:
            p1, p2 = random.choices(top, weights=weights, k=2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)
        self.generations[self.genCount] = new_pop
        self.genCount += 1
        print(f"Best Eval: {scores[0][1]}")
        return scores[0][0]


def save_ga_state(world, filename="ga_state.pkl"):
    state = {
        "genCount": world.genCount,
        "population": world.generations.get(world.genCount - 1, []),
        "population_size": world.size,
        "timestamp": datetime.now(),
        "seed": random.getstate(),
    }
    with open(filename, "wb") as f:
        pickle.dump(state, f)
    print(f"GA state saved to {filename}")


def load_ga_state(filename="ga_state.pkl"):
    with open(filename, "rb") as f:
        state = pickle.load(f)
    print(f"GA state loaded from {filename}")
    return state


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


def load_ga_model_parameters(filename="trained_ga_model.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None


def compute_personal_trainer_move(chess_board):
    best_move = None
    moves = list(chess_board.legal_moves)
    if not moves:
        return None
    best_eval = -float("inf") if chess_board.turn == chess.WHITE else float("inf")
    for move in moves:
        chess_board.push(move)
        t = board_to_tensor(chess_board).unsqueeze(0)
        with torch.no_grad():
            score = trained_model(t).item()
        chess_board.pop()
        if chess_board.turn == chess.WHITE:
            if score > best_eval:
                best_eval = score
                best_move = move
        else:
            if score < best_eval:
                best_eval = score
                best_move = move
    return best_move


def play_human_vs_bot(mode="personal_trainer"):
    if mode == "stockfish":
        try:
            STOCKFISH_PATH = r"C:\Users\Krishna Bansal\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except Exception as e:
            print("Error initializing Stockfish:", e)
            return
        current_board = chess.Board()  # Use standard chess board initialization
    else:
        # For personal trainer mode, use a standard chess board
        current_board = chess.Board()

    human_color = input("Choose your color (w/b): ").strip().lower()
    human_is_white = human_color == "w"
    print("\nStarting game. Enter moves in UCI format (e.g., e2e4).\n")

    while not current_board.is_game_over():
        print(current_board)

        if current_board.turn == (chess.WHITE if human_is_white else chess.BLACK):
            move_uci = input("Your move: ").strip()
            try:
                move = chess.Move.from_uci(move_uci)
                if move in current_board.legal_moves:
                    current_board.push(move)
                else:
                    print("Illegal move. Try again.")
                    continue
            except ValueError:
                print("Invalid move format. Try again.")
                continue
        else:
            print("Bot is thinking...")
            if mode == "stockfish":
                result = engine.play(current_board, chess.engine.Limit(time=0.1))
                if result.move is not None:
                    current_board.push(result.move)
                else:
                    print("Stockfish did not return a valid move.")
            else:
                bot_move = compute_personal_trainer_move(current_board)
                if bot_move:
                    current_board.push(bot_move)
                else:
                    print("Bot has no valid move.")
        print()

    print("Game over!")
    print("Result:", current_board.result())
    if mode == "stockfish":
        engine.quit()


def main():
    while True:
        print("\n1. New GA Evolution\n2. Resume GA Evolution\n3. Play Game\n4. Quit")
        choice = input("Choice: ").strip()
        if choice == "1":
            try:
                gens = int(input("Generations: ").strip())
            except:
                gens = 100
            world = World(5)
            best_board = None
            for _ in range(gens):
                best_board = world.runGeneration()
            if best_board:
                save_ga_state(world)
                save_ga_model(best_board)
        elif choice == "2":
            try:
                state = load_ga_state()
            except:
                print("No saved state.")
                continue
            world = World(
                state["population_size"],
                initial_population=state["population"],
                initial_gen_count=state["genCount"],
            )
            try:
                gens = int(input("Generations to resume: ").strip())
            except:
                gens = 100
            best_board = None
            for _ in range(gens):
                best_board = world.runGeneration()
            if best_board:
                save_ga_state(world)
                save_ga_model(best_board)
        elif choice == "3":
            print("\n1. Stockfish\n2. Personalized Trainer (GA)")
            sub = input("Choice: ").strip()
            if sub == "1":
                play_human_vs_bot("stockfish")
            elif sub == "2":
                play_human_vs_bot("personal_trainer")
            else:
                print("Invalid.")
        elif choice == "4":
            break
        else:
            print("Invalid.")


if __name__ == "__main__":
    main()
