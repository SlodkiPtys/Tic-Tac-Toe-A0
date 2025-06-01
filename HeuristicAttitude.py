import pygame
import sys
import random

# Konfiguracja
BOARD_SIZE = 10
CELL_SIZE = 60
WIN_LENGTH = 5
WIDTH = HEIGHT = BOARD_SIZE * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_COLOR = (50, 50, 50)
X_COLOR = (200, 0, 0)
O_COLOR = (0, 0, 200)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe 10x10")
font = pygame.font.SysFont(None, CELL_SIZE)

board = [['' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
matrix = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

current_player = 'O'
starting_player = 'X'

game_over = False
move_was_not_clicked = True
type_game = 'Bo'

# ---------------------------------------------------------------------------
# Simple heuristic bot for 10×10 Gomoku-style Tic-Tac-Toe (five-in-a-row)
# ---------------------------------------------------------------------------

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]
WEIGHTS    = {4: 100_000,        # win or block-win next move
              3:   1_000,
              2:     100,
              1:      10,
              0:       1}        # fallback when nothing special is found


def _count_in_dir(r, c, dx, dy, player):
    """Return how many consecutive stones <player> has starting NEXT to (r,c)."""
    cnt = 0
    r += dy
    c += dx
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
        cnt += 1
        r += dy
        c += dx
    return cnt


def _score_cell(r, c, player):
    """
    Heuristic score for putting <player> at empty cell (r,c).
    A very large score (>= 100_000) means an immediate win or required block.
    """
    best = 0
    for dx, dy in DIRECTIONS:
        a = _count_in_dir(r, c, dx,  dy, player)
        b = _count_in_dir(r, c, -dx, -dy, player)
        total = a + b
        if total + 1 >= WIN_LENGTH:          # +1 for the stone we are about to place
            return WEIGHTS[4]               # winning / blocking move
        best = max(best, WEIGHTS.get(total, 0))
    return best


def bot_move():
    """Pick the empty cell with the highest heuristic score."""
    global current_player

    opponent = 'O' if current_player == 'X' else 'X'
    best_score = -1
    best_moves = []

    empty_cells = [(r, c) for r in range(BOARD_SIZE)
                           for c in range(BOARD_SIZE) if board[r][c] == '']

    if not empty_cells:                       # no legal move
        return

    for r, c in empty_cells:
        # two-ply heuristic: favour own chances 2× stronger than blocking
        score = (_score_cell(r, c, current_player) * 2 +
                 _score_cell(r, c, opponent))
        if score > best_score:
            best_score = score
            best_moves = [(r, c)]
        elif score == best_score:             # keep ties to randomise a bit
            best_moves.append((r, c))

    r, c = random.choice(best_moves)          # break ties randomly
    board[r][c] = current_player
    game_to_matrix(r, c)

    # check end-of-game conditions
    if check_winner(r, c, current_player):
        draw_board(); pygame.display.update()
        display_winner(f"{current_player} wygrywa!")
        current_player = opponent
        reset_game()
    elif is_draw():
        draw_board(); pygame.display.update()
        display_winner("Remis!")
        current_player = opponent
        reset_game()
    else:                                     # switch sides
        current_player = opponent

def draw_board():
    screen.fill(WHITE)
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (0, y), (WIDTH, y))

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            mark = board[row][col]
            if mark:
                color = X_COLOR if mark == 'X' else O_COLOR
                text = font.render(mark, True, color)
                rect = text.get_rect(center=(col * CELL_SIZE + CELL_SIZE // 2,
                                             row * CELL_SIZE + CELL_SIZE // 2))
                screen.blit(text, rect)

def get_cell(pos):
    x, y = pos
    return y // CELL_SIZE, x // CELL_SIZE

def check_winner(row, col, player):
    def count(dx, dy):
        r, c = row + dy, col + dx
        count = 0
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            count += 1
            r += dy
            c += dx
        return count

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        total = 1 + count(dx, dy) + count(-dx, -dy)
        if total >= WIN_LENGTH:
            return True
    return False

def is_draw():
    return all(cell != '' for row in board for cell in row)

def reset_game():
    global board, current_player, game_over, move_was_not_clicked
    board = [['' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    move_was_not_clicked = True
    game_over = False

def display_winner(text):
    global game_over
    game_over = True
    print(text)  # dla debugowania w konsoli
    pygame.time.wait(100)
    screen.fill(WHITE)
    msg = font.render(text, True, BLACK)
    rect = msg.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(msg, rect)
    pygame.display.update()
    pygame.time.wait(1000)

def print_matrix():
    print("\nActual state of matrix:")
    for row in matrix:
        print(" ".join(str(cell) for cell in row))

def game_to_matrix(row, col):
    board[row][col]
    for col in range(BOARD_SIZE):
        for row in range(BOARD_SIZE):
            if board[row][col] == '':
                matrix[row][col] = 0
            elif board[row][col] == 'X':
                matrix[row][col] = 1
            else:
                matrix[row][col] = 2

    print_matrix()


while True:
    draw_board()
    pygame.display.update()

    if type_game == 'Bot':

        if move_was_not_clicked and starting_player != current_player:
            current_player = 'X'
            bot_move()
            draw_board()
            pygame.display.update()
            move_was_not_clicked = False

        bot_move()
        draw_board()
        pygame.display.update()

    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if game_over:
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                 row, col = get_cell(event.pos)
                 if board[row][col] == '':
                    board[row][col] = current_player
                    game_to_matrix(row, col)

                    if check_winner(row, col, current_player):
                        draw_board()
                        pygame.display.update()
                        display_winner(f"{current_player} wygrywa!")
                        reset_game()
                    elif is_draw():
                        draw_board()
                        pygame.display.update()
                        display_winner("Remis!")
                        reset_game()
                    else:
                          current_player = 'O' if current_player == 'X' else 'X'
                          bot_move()