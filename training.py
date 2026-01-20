import chess
import torch
import torch.nn as nn
import random
from model import ChessNet, board_to_tensor

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL ---
model = ChessNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# --- REPLAY BUFFER ---
replay_buffer = []
MAX_BUFFER = 20_000
BATCH_SIZE = 64

# --- MATERIAL VALUES ---
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 2.97,
    chess.BISHOP: 3.13,
    chess.ROOK: 5.63,
    chess.QUEEN: 9.5,
}

# --- MATERIAL ---
def material_score(board):
    score = 0.0
    for piece, value in PIECE_VALUES.items():
        score += len(board.pieces(piece, chess.WHITE)) * value
        score -= len(board.pieces(piece, chess.BLACK)) * value
    return score

def normalize_material(score):
    return max(-1.0, min(1.0, score / 10.0))

# --- MOBILITY ---
def mobility_score(board):
    white_attacks = set()
    black_attacks = set()

    for sq, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            white_attacks.update(board.attacks(sq))
        else:
            black_attacks.update(board.attacks(sq))

    return float(len(white_attacks) - len(black_attacks))

def normalize_mobility(score):
    return max(-1.0, min(1.0, score / 30.0))

# --- REPETITION PENALTY ---
def repetition_penalty(board):
    return -0.2 if board.is_repetition(2) else 0.0

# --- SAME-PIECE SHUFFLING PENALTY ---
def same_piece_penalty(board):
    if len(board.move_stack) < 2:
        return 0.0

    last = board.move_stack[-1]
    prev = board.move_stack[-2]

    return -0.1 if last.from_square == prev.to_square else 0.0

# --- POSITION TARGET (WHITE-CENTRIC) ---
def position_target(board):
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0

    material = normalize_material(material_score(board))
    mobility = normalize_mobility(mobility_score(board))
    repeat = repetition_penalty(board)
    shuffle = same_piece_penalty(board)

    value = 0.7 * material + 0.3 * mobility + repeat + shuffle
    return max(-1.0, min(1.0, value))

# --- MOVE SELECTION ---
def choose_move(board, model, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(list(board.legal_moves))

    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")
    best_move = None

    for move in board.legal_moves:
        board_copy = board.copy()
        board_copy.push(move)

        input_tensor = board_to_tensor(board_copy).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = model(input_tensor)
            score = value.item()

        if board.turn == chess.WHITE:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move

    return best_move if best_move else random.choice(list(board.legal_moves))

# --- TRAINING LOOP ---
NUM_GAMES = 500
MAX_GAME_LEN = 100

for game in range(NUM_GAMES):
    board = chess.Board()
    history = []

    # --- SELF PLAY ---
    while not board.is_game_over(claim_draw=True) and len(history) < MAX_GAME_LEN:
        board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        target = position_target(board)

        # store WITHOUT turn-based sign flipping
        history.append((board_tensor.clone(), target))

        move = choose_move(board, model, epsilon=0.15)
        board.push(move)

    # --- STORE ---
    for bt, tgt in history:
        replay_buffer.append((bt.cpu(), tgt))
        if len(replay_buffer) > MAX_BUFFER:
            replay_buffer.pop(0)

    # --- TRAIN ---
    if len(replay_buffer) >= BATCH_SIZE:
        batch = random.sample(replay_buffer, BATCH_SIZE)

        inputs = torch.cat([bt for bt, _ in batch]).to(device)
        targets = torch.tensor(
            [tgt for _, tgt in batch],
            dtype=torch.float32
        ).unsqueeze(1).to(device)

        _, values = model(inputs)
        loss = loss_fn(values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Game {game} | "
        f"End: {'Mate' if board.is_checkmate() else 'Draw'} | "
        f"Buffer: {len(replay_buffer)}"
    )

# --- SAVE ---
torch.save(model.state_dict(), "model.pth")
print("Model saved.")
