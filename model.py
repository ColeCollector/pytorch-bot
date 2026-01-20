import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)  # 12 channels for the 12 possible pieces
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        
        # Policy head (moves)
        self.policy_fc = nn.Linear(1024, 4672)  # There are 4672 possible moves on a chessboard
        # Value head (win/lose/draw)
        self.value_fc = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        
        # Policy head: output move probabilities
        policy = self.policy_fc(x)
        
        # Value head: output the value (win/lose/draw)
        value = self.value_fc(x)
        
        return policy, value

# --- Board to Tensor ---
def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row = square // 8
        col = square % 8
        offset = piece.piece_type - 1
        if piece.color == chess.BLACK:
            offset += 6
        tensor[offset][row][col] = 1
    return tensor

# --- Choose the best move from current position ---
def choose_move(board, model):
    best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
    best_move = None

    for move in board.legal_moves:
        board_copy = board.copy()
        board_copy.push(move)

        input_tensor = board_to_tensor(board_copy).unsqueeze(0)

        with torch.no_grad():
            _, value = model(input_tensor)
            score = value.item()

        if board.turn == chess.WHITE and score > best_score:
            best_score = score
            best_move = move
        elif board.turn == chess.BLACK and score < best_score:
            best_score = score
            best_move = move

    return best_move
