import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine
import numpy as np
import os

# -------------------------
# Stockfish Engine Settings
# -------------------------
STOCKFISH_PATH = r"C:\Users\Krishna Bansal\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
try:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
except FileNotFoundError:
    raise SystemExit("❌ ERROR: Stockfish engine not found! Check STOCKFISH_PATH.")


# -------------------------
# Improved Board Encoding
# -------------------------
def board_to_tensor(board):
    """
    Converts a python-chess board to a tensor of shape (12, 8, 8).
    There are 12 channels corresponding to:
      0: White Pawn, 1: White Knight, 2: White Bishop, 3: White Rook,
      4: White Queen, 5: White King,
      6: Black Pawn, 7: Black Knight, 8: Black Bishop, 9: Black Rook,
      10: Black Queen, 11: Black King.
    """
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
        row = 7 - (square // 8)  # convert to row index (0 at bottom)
        col = square % 8
        if piece.color == chess.WHITE:
            channel = piece_to_idx[piece.piece_type]
        else:
            channel = piece_to_idx[piece.piece_type] + 6
        tensor[channel, row, col] = 1.0
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)  # shape: (1,12,8,8)


# -------------------------
# Improved Custom Chess Model (CNN)
# -------------------------
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
        self.fc2 = nn.Linear(512, 1)  # Output evaluation score

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


# -------------------------
# Stockfish Evaluation Function
# -------------------------
def get_stockfish_eval(board_fen):
    """
    Uses Stockfish to evaluate a board position given as a FEN string.
    Returns the evaluation normalized (centipawn value divided by 100).
    """
    board = chess.Board(board_fen)
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=12))
        score = info["score"].relative.score(mate_score=1000)
        if score is None:
            return 0.0
        return score / 100.0
    except Exception as e:
        print(f"⚠️ Warning: Stockfish evaluation failed for {board_fen}: {e}")
        return 0.0


# -------------------------
# Training Function
# -------------------------
def train_model(model, training_data, epochs=1000, lr=0.001):
    """
    Trains the CustomChessModel using a list of FEN strings as training data.
    The target evaluation for each position is obtained from Stockfish.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for board_fen in training_data:
            target_eval = get_stockfish_eval(board_fen)
            board = chess.Board(board_fen)
            input_tensor = board_to_tensor(board)  # shape: (1,12,8,8)

            optimizer.zero_grad()
            output = model(input_tensor)
            target_tensor = torch.tensor([[target_eval]], dtype=torch.float32)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    model.eval()


# -------------------------
# Training Data (Example FENs)
# -------------------------
training_data = [
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    # For best results, include many more FEN positions from actual games or databases.
]

# -------------------------
# Model Initialization and Training
# -------------------------
custom_model = CustomChessModel()
MODEL_PATH = "custom_stockfish_model.pth"

if os.path.exists(MODEL_PATH):
    print("✅ Model already trained. Loading existing weights...")
    custom_model.load_state_dict(torch.load(MODEL_PATH))
else:
    print("🚀 Training model from scratch...")
    train_model(custom_model, training_data, epochs=1000, lr=0.001)
    torch.save(custom_model.state_dict(), MODEL_PATH)
    print("✅ Model trained and saved.")

# -------------------------
# Clean Up
# -------------------------
engine.quit()
print("✨ Done!")
