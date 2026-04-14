import pygame
import chess
import torch
from model import ChessNet, choose_move

# --- INIT PYGAME ---
pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Chess Bot")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 64)
small_font = pygame.font.SysFont(None, 32)

# --- LOAD IMAGES ---
PIECE_IMAGES = {}

for color in ["white", "black"]:
    for name in ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]:
        abbrev = "N" if name == "Knight" else name[0]
        color_key = "w" if color == "white" else "b"
        key = color_key + abbrev
        path = f"images/{color}/{name}.png"
        PIECE_IMAGES[key] = pygame.transform.scale(
            pygame.image.load(path), (60, 60)
        )

# --- DRAW BOARD ---
def draw_board(screen, board):
    screen.fill("white")
    colors = ["white", "#D684FF"]
    selected_img = None

    for rank in range(8):
        for file in range(8):

            if player_color == chess.WHITE:
                display_file = file
                display_rank = 7 - rank
            else:
                display_file = 7 - file
                display_rank = rank

            pygame.draw.rect(
                screen,
                colors[(rank + file) % 2],
                pygame.Rect(
                    360 + display_file * 60,
                    60 + display_rank * 60,
                    60,
                    60,
                ),
            )

            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece:
                abbrev = piece.symbol().upper()[0]
                color = "w" if piece.color == chess.WHITE else "b"
                img = PIECE_IMAGES[color + abbrev]

                if square == selected_square:
                    selected_img = img
                else:
                    screen.blit(
                        img,
                        pygame.Rect(
                            360 + display_file * 60,
                            60 + display_rank * 60,
                            60,
                            60,
                        ),
                    )

    if selected_img:
        mx, my = pygame.mouse.get_pos()
        screen.blit(selected_img, (mx - 30, my - 30))


# --- END SCREEN ---
def draw_end_screen(screen, board):
    if board.is_checkmate():
        text = "You Won!" if board.turn != player_color else "You Lost!"
    else:
        text = "Draw"

    overlay = pygame.Surface((1280, 720), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    screen.blit(font.render(text, True, (255, 255, 255)), (520, 300))
    screen.blit(
        small_font.render("Click anywhere to play again", True, (200, 200, 200)),
        (470, 380),
    )


def draw_side_select():
    screen.fill("#0b0b0e")
    mouse_pos = pygame.mouse.get_pos()

    title = font.render("Choose Your Side", True, (255, 255, 255))
    screen.blit(title, title.get_rect(center=(640, 200)))

    white_rect = pygame.Rect(440, 350, 180, 80)
    black_rect = pygame.Rect(660, 350, 180, 80)

    white_color = (
        (200, 200, 200) if white_rect.collidepoint(mouse_pos) else (230, 230, 230)
    )
    black_color = (
        (70, 70, 70) if black_rect.collidepoint(mouse_pos) else (40, 40, 40)
    )

    pygame.draw.rect(screen, white_color, white_rect, border_radius=8)
    pygame.draw.rect(screen, black_color, black_rect, border_radius=8)

    screen.blit(
        small_font.render("Play White", True, (0, 0, 0)),
        small_font.render("Play White", True, (0, 0, 0)).get_rect(center=white_rect.center),
    )
    screen.blit(
        small_font.render("Play Black", True, (255, 255, 255)),
        small_font.render("Play Black", True, (255, 255, 255)).get_rect(center=black_rect.center),
    )

    return white_rect, black_rect


def get_square_under_mouse():
    x, y = pygame.mouse.get_pos()
    file = (x - 360) // 60
    rank = (y - 60) // 60

    if not (0 <= file <= 7 and 0 <= rank <= 7):
        return None

    if player_color == chess.WHITE:
        board_file = file
        board_rank = 7 - rank
    else:
        board_file = 7 - file
        board_rank = rank

    return chess.square(board_file, board_rank)


# --- PROMOTION ---
def create_move_with_promotion(board, from_square, to_square):
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(to_square)
        if (piece.color == chess.WHITE and rank == 7) or (
            piece.color == chess.BLACK and rank == 0
        ):
            return chess.Move(from_square, to_square, promotion=chess.QUEEN)
    return chess.Move(from_square, to_square)


# --- GAME STATE ---
board = chess.Board()
selected_square = None
game_over = False
choosing_side = True
player_color = chess.WHITE

model = ChessNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# --- MAIN LOOP ---
running = True
while running:
    if choosing_side:
        white_rect, black_rect = draw_side_select()
    else:
        draw_board(screen, board)
        if game_over:
            draw_end_screen(screen, board)

    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if choosing_side and event.type == pygame.MOUSEBUTTONDOWN:
            if white_rect.collidepoint(event.pos):
                player_color = chess.WHITE
                choosing_side = False
            elif black_rect.collidepoint(event.pos):
                player_color = chess.BLACK
                choosing_side = False
                board.push(choose_move(board, model))

        if game_over and event.type == pygame.MOUSEBUTTONDOWN:
            board = chess.Board()
            selected_square = None
            game_over = False
            choosing_side = True
            continue

        if choosing_side or game_over:
            continue

        if event.type == pygame.MOUSEBUTTONDOWN:
            square = get_square_under_mouse()
            if square is not None:
                if selected_square is None:
                    piece = board.piece_at(square)
                    if piece and piece.color == player_color:
                        selected_square = square
                else:
                    move = create_move_with_promotion(board, selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None

                        if board.is_game_over():
                            game_over = True
                            break

                        board.push(choose_move(board, model))
                        if board.is_game_over():
                            game_over = True
                    else:
                        selected_square = None

pygame.quit()
