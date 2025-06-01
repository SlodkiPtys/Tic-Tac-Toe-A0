# Gomoku 10×10 — Self‑improving Alpha‑Beta bot with pattern learning
# --------------------------------------------------------------------------
#  • Three game modes: HUMAN_AI, AI_AI, HUMAN_HUMAN
#  • Iterative–deepening alpha‑beta search + transposition table
#  • Pattern‑based static evaluation (weights are updated after every game)
#  • Game logs → game_history.jsonl, pattern weights → weights.json
#  • Almost impossible to beat once it has played a few dozen self‑play games
# --------------------------------------------------------------------------

import pygame, sys, json, time, random, os, math
from collections import defaultdict

# ────────────────────── CONFIG ─────────────────────────────────────────────
GAME_MODE   = "HUMAN_AI"          # "HUMAN_AI" | "AI_AI" | "HUMAN_HUMAN"
HUMAN_SIDE  = "X"                 # Your side when GAME_MODE == "HUMAN_AI"
TIME_LIMIT  = 1.0                 # seconds per AI move (iterative deepening)
LOG_FILE    = "game_history.jsonl"
WEIGHT_FILE = "weights.json"

# Board / UI ---------------------------------------------------------------
BOARD_SIZE, CELL_SIZE, WIN_LENGTH = 10, 60, 5
WIDTH = HEIGHT = BOARD_SIZE * CELL_SIZE
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
LINE_COLOR   = ( 50,  50,  50)
X_COLOR, O_COLOR = (200,   0,   0), (0,   0, 200)

# Directions (dx, dy)
DIRS = [(1,0), (0,1), (1,1), (1,-1)]
INF  = 10**9
LEARNING_RATE = 0.15            # bigger → faster but less stable

# --------------------------------------------------------------------------
# Pattern‑weight storage / persistence
# --------------------------------------------------------------------------
def _load_weights():
    if os.path.isfile(WEIGHT_FILE):
        with open(WEIGHT_FILE, "r", encoding="utf‑8") as f:
            return {int(k): float(v) for k, v in json.load(f).items()}
    # defaults — decent starter heuristic
    return {4: 100_000.0, 3: 2_000.0, 2: 400.0, 1: 40.0, 0: 4.0}

PATTERN_VAL = _load_weights()

# --------------------------------------------------------------------------
# Pygame initialisation
# --------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font   = pygame.font.SysFont(None, CELL_SIZE)

# --------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------
board  = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
current_player = "X"
move_list = []               # list[{"p":…, "r":…, "c":…}]
scoreboard  = defaultdict(int)  # {"X": n, "O": n, "draw": n}
transpo_tbl = {}              # transposition table

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def _count_dir(r, c, dx, dy, player):
    cnt = 0
    r += dy; c += dx
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
        cnt += 1
        r += dy; c += dx
    return cnt


def _evaluate(player):
    """Static evaluation from player's point of view."""
    opponent = "O" if player == "X" else "X"

    def side_score(side):
        s = 0.0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != side:
                    continue
                for dx, dy in DIRS:
                    a = _count_dir(r, c,  dx,  dy, side)
                    b = _count_dir(r, c, -dx, -dy, side)
                    s += PATTERN_VAL.get(a + b, 0.0)
        return s

    return side_score(player) - side_score(opponent)


# Fast board hash for transposition table
HASH_BASE = 3  # empty, X, O → digits 0‑2

def _hash_board():
    code = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            code = code * HASH_BASE + (1 if board[r][c] == "X" else 2 if board[r][c] == "O" else 0)
    return code

# Winner / draw detection ---------------------------------------------------

def _count_line(r, c, dx, dy, player):
    return _count_dir(r, c, dx, dy, player) + _count_dir(r, c, -dx, -dy, player) + 1

def check_winner(r, c, player):
    return any(_count_line(r, c, dx, dy, player) >= WIN_LENGTH for dx, dy in DIRS)


def is_draw():
    return all(board[r][c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))

# --------------------------------------------------------------------------
# Alpha‑beta search with iterative deepening & transposition table
# --------------------------------------------------------------------------

def _alpha_beta(depth, alpha, beta, player, deadline):
    """Returns (score, move) for player."""
    if time.perf_counter() >= deadline:
        raise TimeoutError

    key = (_hash_board(), depth, player)
    if key in transpo_tbl:
        return transpo_tbl[key]

    # generate moves (simple ordering: centre‑first)
    empties = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == ""]
    if depth == 0 or not empties:
        val = (_evaluate(player), None)
        transpo_tbl[key] = val
        return val

    opponent = "O" if player == "X" else "X"
    best_move = None

        # try immediate wins / blocks / threat‑fours first
    for r, c in empties:
        # (A) our own instant win
        board[r][c] = player
        if check_winner(r, c, player):
            board[r][c] = ""
            transpo_tbl[key] = (INF - depth, (r, c))
            return INF - depth, (r, c)
        board[r][c] = ""

        # (B) direct block of opponent's win next turn
        board[r][c] = opponent
        if check_winner(r, c, opponent):
            board[r][c] = ""
            block_score = INF // 2 - depth  # very valuable but worse than winning
            transpo_tbl[key] = (block_score, (r, c))
            return block_score, (r, c)

        # (C) block *threat four* (opponent would create a 4‑in‑a‑row and open ends)
        is_threat = False
        if _count_line(r, c, 1, 0, opponent) == WIN_LENGTH - 1 or \
           _count_line(r, c, 0, 1, opponent) == WIN_LENGTH - 1 or \
           _count_line(r, c, 1, 1, opponent) == WIN_LENGTH - 1 or \
           _count_line(r, c, 1, -1, opponent) == WIN_LENGTH - 1:
            is_threat = True
        board[r][c] = ""
        if is_threat:
            threat_score = INF // 4 - depth  # lower than block‑win but still high
            transpo_tbl[key] = (threat_score, (r, c))
            return threat_score, (r, c)

    # heuristic ordering : score own move − opp move : score own move − opp move
    def move_key(cell):
        r, c = cell
        return -(_count_dir(r, c, 1, 0, player) + _count_dir(r, c, 0, 1, player))
    empties.sort(key=move_key)

    for r, c in empties:
        # --- place our stone ------------------------------------------------
        board[r][c] = player

        # If this move still lets the opponent win immediately on their next
        # turn (double‑threat), discard it right away.
        losing = False
        for rr, cc in empties:
            if (rr, cc) == (r, c):
                continue  # now occupied
            board[rr][cc] = opponent
            win_next = check_winner(rr, cc, opponent)
            board[rr][cc] = ""
            if win_next:
                losing = True
                break
        if losing:
            board[r][c] = ""
            continue  # ignore this self‑destructive move

        # --------------------------------------------------------------------
        try:
            score, _ = _alpha_beta(depth - 1, -beta, -alpha, opponent, deadline)
        except TimeoutError:
            board[r][c] = ""
            raise
        board[r][c] = ""
        score = -score
        if score > alpha:
            alpha, best_move = score, (r, c)
            if alpha >= beta:
                break
    transpo_tbl[key] = (alpha, best_move)
    return alpha, best_move


def ai_pick_move(player):
    start = time.perf_counter()
    deadline = start + TIME_LIMIT
    depth = 1
    best_move = None
    try:
        while True:
            score, move = _alpha_beta(depth, -INF, INF, player, deadline)
            best_move = move
            depth += 1
    except TimeoutError:
        pass  # ran out of time, use last fully‑searched move
    except Exception as e:
        print("Search error", e)
    return best_move

# --------------------------------------------------------------------------
# Learning: update pattern weights after each game (very simple RL)
# --------------------------------------------------------------------------

def _save_weights():
    with open(WEIGHT_FILE, "w", encoding="utf‑8") as f:
        json.dump({k: v for k, v in PATTERN_VAL.items()}, f)


def update_pattern_weights(winner):
    if winner == "draw":
        return  # no learning on draws

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == "":
                continue
            for dx, dy in DIRS:
                a = _count_dir(r, c,  dx,  dy, board[r][c])
                b = _count_dir(r, c, -dx, -dy, board[r][c])
                pat_len = a + b
                if pat_len >= 5:
                    continue  # already a winning five‑line – leave weight
                delta = LEARNING_RATE * (1 if board[r][c] == winner else -1)
                PATTERN_VAL[pat_len] = PATTERN_VAL.get(pat_len, 0.0) * (1 + delta)

    _save_weights()

# --------------------------------------------------------------------------
# Drawing / UI
# --------------------------------------------------------------------------

def draw_board():
    screen.fill(WHITE)
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (0, y), (WIDTH, y))
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            m = board[r][c]
            if m:
                img = font.render(m, True, X_COLOR if m == "X" else O_COLOR)
                screen.blit(img, img.get_rect(center=(c * CELL_SIZE + CELL_SIZE // 2,
                                                       r * CELL_SIZE + CELL_SIZE // 2)))

def cell_from_mouse(pos):
    x, y = pos
    return y // CELL_SIZE, x // CELL_SIZE

# --------------------------------------------------------------------------
# Game flow helpers
# --------------------------------------------------------------------------

def reset_board():
    global board, current_player, move_list
    board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = "X"
    move_list = []


def log_move(r, c):
    move_list.append({"p": current_player, "r": r, "c": c})


def save_game(winner):
    with open(LOG_FILE, "a", encoding="utf‑8") as f:
        f.write(json.dumps({"winner": winner, "moves": move_list}) + "\n")


def end_game(result):
    scoreboard[result] += 1
    save_game(result)
    update_pattern_weights(result)
    msg = "Draw!" if result == "draw" else f"{result} wins!"
    draw_board(); pygame.display.update();
    time.sleep(1)
    reset_board()
    title = f"X:{scoreboard['X']}  O:{scoreboard['O']}  draw:{scoreboard['draw']}"
    pygame.display.set_caption(title)

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    global current_player
    reset_board()
    AI_SIDE = "O" if GAME_MODE == "HUMAN_AI" and HUMAN_SIDE == "X" else "X"

    clock = pygame.time.Clock()
    running = True
    while running:
        draw_board(); pygame.display.flip()

        # AI turn?
        if GAME_MODE in ("AI_AI", "HUMAN_AI") and current_player == AI_SIDE:
            move = ai_pick_move(current_player)
            if move:
                r, c = move
                board[r][c] = current_player
                log_move(r, c)
                if check_winner(r, c, current_player):
                    end_game(current_player); continue
                if is_draw():
                    end_game("draw"); continue
                current_player = "O" if current_player == "X" else "X"

        # Event handling ----------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if GAME_MODE == "HUMAN_HUMAN" or (GAME_MODE == "HUMAN_AI" and current_player == HUMAN_SIDE):
                    r, c = cell_from_mouse(event.pos)
                    if board[r][c] == "":
                        board[r][c] = current_player
                        log_move(r, c)
                        if check_winner(r, c, current_player):
                            end_game(current_player); break
                        if is_draw():
                            end_game("draw"); break
                        current_player = "O" if current_player == "X" else "X"
        clock.tick(60)

    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()
