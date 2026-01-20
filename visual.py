import chess
import torch
import torch.nn as nn
import random
from model import ChessNet, board_to_tensor

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
model = ChessNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ---------------- REPLAY BUFFER ----------------
replay_buffer = []
MAX_BUFFER = 20_000
BATCH_SIZE = 64

# ---------------- MATERIAL VALUES ----------------
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}

# ---------------- EVALUATION ----------------
def material_score(board):
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score

def normalize(x):
    return max(-1.0, min(1.0, x))

def position_target(board):
    # Checkmate
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0

    score = material_score(board) / 10.0

    # Encourage development (minor pieces off back rank)
    score += 0.05 * (
        len(board.pieces(chess.KNIGHT, chess.WHITE)) +
        len(board.pieces(chess.BISHOP, chess.WHITE))
    )
    score -= 0.05 * (
        len(board.pieces(chess.KNIGHT, chess.BLACK)) +
        len(board.pieces(chess.BISHOP, chess.BLACK))
    )

    # Small penalty for being in check
    if board.is_check():
        score -= 0.1 if board.turn == chess.WHITE else -0.1

    # MOVE PENALTY (prevents shuffling)
    score -= 0.01

    return normalize(score)

# ---------------- MOVE SELECTION ----------------
def choose_move(board, model, epsilon):
    if random.random() < epsilon:
        return random.choice(list(board.legal_moves))

    best_value = -float("inf") if board.turn == chess.WHITE else float("inf")
    best_move = None

    for move in board.legal_moves:
        b = board.copy()
        b.push(move)

        inp = board_to_tensor(b).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = model(inp)
            v = value.item()

        if board.turn == chess.WHITE and v > best_value:
            best_value = v
            best_move = move
        elif board.turn == chess.BLACK and v < best_value:
            best_value = v
            best_move = move

    return best_move if best_move else random.choice(list(board.legal_moves))

# ---------------- TRAINING LOOP ----------------
NUM_GAMES = 500
MAX_GAME_LEN = 100

for game in range(NUM_GAMES):
    board = chess.Board()
    history = []

    epsilon = max(0.05, 1.0 - game / 300)

    # -------- SELF PLAY --------
    while not board.is_game_over(claim_draw=True) and len(history) < MAX_GAME_LEN:
        board_tensor = board_to_tensor(board).unsqueeze(0)

        turn = board.turn
        move = choose_move(board, model, epsilon)
        board.push(move)

        target = position_target(board)
        history.append((board_tensor, turn, target))

    # -------- FINAL GAME RESULT --------
    if board.is_checkmate():
        final_reward = 1.0 if board.turn == chess.BLACK else -1.0
    else:
        # DRAW = BAD
        final_reward = -0.2

    # Apply final reward to all states
    for i in range(len(history)):
        bt, turn, _ = history[i]
        history[i] = (bt, turn, final_reward)

    # -------- STORE IN REPLAY BUFFER --------
    for board_tensor, turn, target in history:
        replay_buffer.append((board_tensor, turn, target))
        if len(replay_buffer) > MAX_BUFFER:
            replay_buffer.pop(0)

    # -------- TRAIN --------
    if len(replay_buffer) >= BATCH_SIZE:
        batch = random.sample(replay_buffer, BATCH_SIZE)

        inputs = []
        targets = []

        for board_tensor, turn, target in batch:
            inputs.append(board_tensor)
            targets.append(target if turn == chess.WHITE else -target)

        inputs = torch.cat(inputs).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)

        _, values = model(inputs)
        loss = loss_fn(values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Game {game} | "
        f"Epsilon {epsilon:.2f} | "
        f"Result: {'Checkmate' if board.is_checkmate() else 'Draw'} | "
        f"Replay: {len(replay_buffer)}"
    )

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "model.pth")
print("Model saved.")
