import pygame, sys, random, json, os, time
# ── GLOBAL SETTINGS ─────────────────────────────────────────────────────────
GAME_MODE  = "HUMAN_AI"     # "HUMAN_AI" | "AI_AI" | "HUMAN_HUMAN"
HUMAN_SIDE = "X"            # your side when GAME_MODE == "HUMAN_AI"
MAX_DEPTH  = 3              # AI search depth
LOG_FILE   = "game_history.jsonl"
# ── CONSTANTS ───────────────────────────────────────────────────────────────
BOARD_SIZE , CELL_SIZE , WIN_LENGTH = 10, 60, 5
WIDTH = HEIGHT = BOARD_SIZE * CELL_SIZE
WHITE, BLACK  = (255,255,255), (0,0,0)
LINE_COLOR    = (50,50,50)
X_COLOR, O_COLOR = (200,0,0), (0,0,200)
INF = 10**9
PATTERN_VAL = {4:100_000, 3:2_000, 2:400, 1:40, 0:4}
DIRECTIONS  = [(1,0),(0,1),(1,1),(1,-1)]
# ── PYGAME INIT ─────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font   = pygame.font.SysFont(None, CELL_SIZE)
pygame.display.set_caption("10×10 Five-in-a-Row")
# ── STATE -------------------------------------------------------------------
board  = [['' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
matrix = [[0   for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
current_player, game_over = 'X', False
game_moves, scoreboard    = [], {"X":0, "O":0, "draw":0}
# ── BOT (alpha-beta search) ────────────────────────────────────────────────
def _count_dir(r, c, dx, dy, player):
    """Count consecutive stones belonging to *player* starting
    from the square next to (r, c) in the (dx, dy) direction."""
    count = 0
    r += dy          # step once to the neighbouring square
    c += dx
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
        count += 1
        r += dy
        c += dx
    return count

def _evaluate(player):
    """Static evaluation of the current board from *player*’s point of view."""
    opponent = 'O' if player == 'X' else 'X'

    def side_score(side):
        total = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != side:
                    continue
                for dx, dy in DIRECTIONS:
                    a = _count_dir(r, c,  dx,  dy, side)
                    b = _count_dir(r, c, -dx, -dy, side)
                    total += PATTERN_VAL.get(a + b, 0)
        return total

    return side_score(player) - side_score(opponent)

def _alpha_beta(d,a,b,p):
    o='O'if p=='X'else'X'
    free=[(r,c)for r in range(BOARD_SIZE)for c in range(BOARD_SIZE)if board[r][c]=='']
    if d==0 or not free: return _evaluate(p),None
    for r,c in free:
        board[r][c]=p
        if check_winner(r,c,p): board[r][c]=''; return INF-d,(r,c)
        board[r][c]=o
        if check_winner(r,c,o): board[r][c]=''; return -INF+d,(r,c)
        board[r][c]=''
    best=None
    for r,c in free:
        board[r][c]=p
        s,_=_alpha_beta(d-1,-b,-a,o); board[r][c]=''; s=-s
        if s>a: a,best=s,(r,c);
        if a>=b: break
    return a,best
def bot_move():
    global current_player
    _,mv=_alpha_beta(MAX_DEPTH,-INF,INF,current_player)
    if not mv:return
    r,c=mv; board[r][c]=current_player; log_move(r,c)
    update_matrix(r,c); end_checks(r,c)
    current_player='O'if current_player=='X'else'X'
# ── GAME LOGGING / LEARNING HOOKS ───────────────────────────────────────────
def log_move(r,c):
    game_moves.append({"p":current_player,"r":r,"c":c})
def save_game(winner):
    entry={"winner":winner,"moves":game_moves}
    with open(LOG_FILE,"a",encoding="utf-8")as f: f.write(json.dumps(entry)+"\n")
def update_pattern_weights(winner):
    """
    Hook for learning. Right now it does *nothing* but you can:
      • read game_moves (list[dict]) and PATTERN_VAL
      • change PATTERN_VAL in-place, e.g. with simple
        reinforcement or whatever scheme you invent.
    """
    pass
# ── UI HELPERS ──────────────────────────────────────────────────────────────
def draw_board():
    screen.fill(WHITE)
    for x in range(0,WIDTH,CELL_SIZE): pygame.draw.line(screen,LINE_COLOR,(x,0),(x,HEIGHT))
    for y in range(0,HEIGHT,CELL_SIZE):pygame.draw.line(screen,LINE_COLOR,(0,y),(WIDTH,y))
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            m=board[r][c]
            if m:
                img=font.render(m,True,X_COLOR if m=='X'else O_COLOR)
                screen.blit(img,img.get_rect(center=(c*CELL_SIZE+CELL_SIZE//2,
                                                     r*CELL_SIZE+CELL_SIZE//2)))
def cell_from_mouse(pos): x,y=pos; return y//CELL_SIZE,x//CELL_SIZE
def check_winner(r,c,p):
    def cnt(dx,dy):
        rr,cc,n=r+dy,c+dx,0
        while 0<=rr<BOARD_SIZE and 0<=cc<BOARD_SIZE and board[rr][cc]==p:
            n+=1; rr+=dy; cc+=dx
        return n
    return any(1+cnt(dx,dy)+cnt(-dx,-dy)>=WIN_LENGTH for dx,dy in DIRECTIONS)
def is_draw(): return all(cell for row in board for cell in row)
def end_checks(r,c):
    global game_over
    if check_winner(r,c,board[r][c]):
        finish_game(board[r][c])
    elif is_draw():
        finish_game("draw")
def finish_game(result):
    global game_over, scoreboard
    scoreboard[result]+=1
    save_game(result); update_pattern_weights(result)
    msg="Draw!" if result=="draw" else f"{result} wins!"
    draw_board(); pygame.display.update(); show_msg(msg)
    game_over=True; reset()
    title=f"{scoreboard['X']}-{scoreboard['O']}-{scoreboard['draw']}  (X-wins : O-wins : draws)"
    pygame.display.set_caption(title)
def show_msg(msg):
    pygame.time.wait(200)
    screen.fill(WHITE); img=font.render(msg,True,BLACK)
    screen.blit(img,img.get_rect(center=(WIDTH//2,HEIGHT//2)))
    pygame.display.update(); pygame.time.wait(1000)
def reset():
    global board,matrix,current_player,game_over,game_moves
    board=[[''for _ in range(BOARD_SIZE)]for _ in range(BOARD_SIZE)]
    matrix=[[0 for _ in range(BOARD_SIZE)]for _ in range(BOARD_SIZE)]
    current_player,game_over='X',False
    game_moves=[]
def update_matrix(r,c): matrix[r][c]=1 if board[r][c]=='X'else 2
# ── MAIN LOOP ───────────────────────────────────────────────────────────────
AI_SIDE='O' if GAME_MODE=="HUMAN_AI" and HUMAN_SIDE=='X' else 'X'
reset()
while True:
    draw_board(); pygame.display.update()
    if not game_over:
        if GAME_MODE=="AI_AI": bot_move()
        elif GAME_MODE=="HUMAN_AI" and current_player==AI_SIDE: bot_move()
    for e in pygame.event.get():
        if e.type==pygame.QUIT: pygame.quit(); sys.exit()
        if game_over: continue
        if e.type==pygame.MOUSEBUTTONDOWN:
            r,c=cell_from_mouse(e.pos)
            if board[r][c]=='':                     # empty cell?
                legal=(GAME_MODE=="HUMAN_HUMAN" or
                       (GAME_MODE=="HUMAN_AI" and current_player==HUMAN_SIDE))
                if legal:
                    board[r][c]=current_player; log_move(r,c)
                    update_matrix(r,c); end_checks(r,c)
                    current_player='O'if current_player=='X'else'X'
