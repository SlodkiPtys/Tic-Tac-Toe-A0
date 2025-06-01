"""AlphaZero-style 10×10 Tic-Tac-Toe (persistent model)
Usage:
  python alphazero_tictactoe_10x10.py [--train-iter N] [--model PATH] [--overwrite]
"""
from __future__ import annotations
import numpy as np, tensorflow as tf, math, argparse, os
from dataclasses import dataclass

BOARD_SIZE, WIN_LEN = 10, 5
EMPTY, X, O = 0, 1, -1

class Env:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)
        self.cur, self.moves, self.done, self.winner = X, 0, False, None
    def legal(self):
        return [i for i in range(BOARD_SIZE**2) if self.board.flat[i] == EMPTY]
    def _line(self, r, c, dr, dc):
        cnt = 0; p = self.board[r, c]
        for k in range(-WIN_LEN + 1, WIN_LEN):
            rr, cc = r + dr * k, c + dc * k
            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and self.board[rr, cc] == p:
                cnt += 1
                if cnt >= WIN_LEN:
                    return True
            else:
                cnt = 0
        return False
    def _win(self, r, c):
        return any(self._line(r, c, dr, dc) for dr, dc in ((1,0),(0,1),(1,1),(1,-1)))
    def state(self):
        cp = np.full((BOARD_SIZE, BOARD_SIZE, 1), self.cur, np.int8)
        own = (self.board == self.cur).astype(np.int8)[..., None]
        opp = (self.board == -self.cur).astype(np.int8)[..., None]
        return np.concatenate([own, opp, cp], -1)
    def step(self, m):
        assert not self.done; r, c = divmod(m, BOARD_SIZE); assert self.board[r, c] == EMPTY
        self.board[r, c] = self.cur; self.moves += 1
        if self._win(r, c): self.done, self.winner = True, self.cur
        elif self.moves == BOARD_SIZE**2: self.done = True
        s = self.state(); self.cur *= -1
        return s, 0.0, self.done, {}

class Heur:
    def pick(self, env: Env):
        for m in env.legal():
            r, c = divmod(m, BOARD_SIZE); env.board[r, c] = env.cur
            if env._win(r, c): env.board[r, c] = EMPTY; return m
            env.board[r, c] = EMPTY
        for m in env.legal():
            r, c = divmod(m, BOARD_SIZE); env.board[r, c] = -env.cur
            if env._win(r, c): env.board[r, c] = EMPTY; return m
            env.board[r, c] = EMPTY
        return -1

def build_model():
    inp = tf.keras.Input((BOARD_SIZE, BOARD_SIZE, 3))
    x = inp
    for _ in range(4):
        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    p = tf.keras.layers.Flatten()(tf.keras.layers.Conv2D(2,1,activation='relu')(x))
    logits = tf.keras.layers.Dense(BOARD_SIZE**2)(p)
    v = tf.keras.layers.Flatten()(tf.keras.layers.Conv2D(1,1,activation='relu')(x))
    v = tf.keras.layers.Dense(64, activation='relu')(v)
    value = tf.keras.layers.Dense(1, activation='tanh')(v)
    return tf.keras.Model(inp, [logits, value])

@dataclass
class Node:
    prior: float; visits: int = 0; value: float = 0.0; children: dict | None = None
    def q(self):
        return 0 if self.visits == 0 else self.value / self.visits

class MCTS:
    def __init__(self, model, sims=200, c=1.4):
        self.model, self.sims, self.c = model, sims, c
    def run(self, env):
        root = Node(1.0, children={}); self._expand(root, env)
        for _ in range(self.sims):
            e = self._copy(env); n = root; path = [n]
            while n.children:
                m, n = self._select(n); e.step(m); path.append(n)
            if not e.done: self._expand(n, e)
            val = self._eval(e)
            for node in path:
                node.visits += 1; node.value += val; val = -val
        π = np.zeros(BOARD_SIZE**2, np.float32)
        for m, ch in root.children.items(): π[m] = ch.visits
        π /= π.sum() if π.sum() else 1
        return π
    def _select(self, n):
        tot = math.sqrt(sum(ch.visits for ch in n.children.values()))
        best, bm, bc = -1e9, -1, None
        for m, ch in n.children.items():
            u = self.c * ch.prior * tot / (1 + ch.visits)
            score = ch.q() + u
            if score > best: best, bm, bc = score, m, ch
        return bm, bc
    def _expand(self, n, env):
        logits, _ = self.model(env.state()[None].astype(np.float32), training=False)
        pri = tf.nn.softmax(logits[0]).numpy()
        n.children = {m: Node(pri[m], children={}) for m in env.legal()}
    def _eval(self, env):
        if env.done:
            if env.winner is None: return 0.0
            return 1.0 if env.winner == env.cur else -1.0
        return float(self.model(env.state()[None].astype(np.float32), training=False)[1][0,0])
    def _copy(self, env):
        e = Env(); e.board = env.board.copy(); e.cur = env.cur; e.moves = env.moves
        e.done, e.winner = env.done, env.winner; return e

class Agent(Heur):
    def __init__(self, model, sims=200): self.mcts = MCTS(model, sims)
    def move(self, env, temp=1.0):
        m = self.pick(env)
        if m != -1: return m
        π = self.mcts.run(env)
        if temp == 0: return int(np.argmax(π))
        π = np.power(π, 1 / temp); π /= π.sum(); return int(np.random.choice(len(π), p=π))

class Learner:
    def __init__(self):
        self.model = build_model(); self.opt = tf.keras.optimizers.Adam(1e-3); self.buf = []
    def self_play(self, n=20):
        for _ in range(n):
            env, agent = Env(), Agent(self.model)
            S, P, Pl = [], [], []
            while not env.done:
                π = agent.mcts.run(env); m = agent.move(env, 1.0)
                S.append(env.state()); P.append(π); Pl.append(env.cur); env.step(m)
            z = 0.0 if env.winner is None else 1.0
            for s, π, p in zip(S, P, Pl): self.buf.append((s.astype(np.float32), π.astype(np.float32), z if p == env.winner else -z))
    def train(self, batch=64):
        if len(self.buf) < batch: return
        idx = np.random.choice(len(self.buf), batch, False)
        s, π, z = zip(*(self.buf[i] for i in idx))
        s, π, z = np.stack(s), np.stack(π), np.array(z, np.float32)
        with tf.GradientTape() as t:
            logits, val = self.model(s, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(π, logits, from_logits=True) + tf.keras.losses.mean_squared_error(z, tf.squeeze(val)))
        self.opt.apply_gradients(zip(t.gradient(loss, self.model.trainable_variables), self.model.trainable_variables))
    def learn(self, iters=100):
        for i in range(iters):
            self.self_play(); self.train()
            if (i+1) % 10 == 0: print(f"Iter {i+1}/{iters} buf {len(self.buf)}")
    def save(self, path): self.model.save_weights(path)
    def load(self, path): self.model.load_weights(path)
    def play(self):
        import pygame
        pygame.init(); CELL = 60; W = H = BOARD_SIZE*CELL
        screen = pygame.display.set_mode((W, H)); font = pygame.font.SysFont(None, CELL)
        env, bot = Env(), Agent(self.model, 200)
        def draw():
            screen.fill((255,255,255))
            for x in range(0,W,CELL): pygame.draw.line(screen,(0,0,0),(x,0),(x,H))
            for y in range(0,H,CELL): pygame.draw.line(screen,(0,0,0),(0,y),(W,y))
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    v = env.board[r,c]
                    if v != EMPTY:
                        t = font.render('X' if v==X else 'O', True, (200,0,0) if v==X else (0,0,200))
                        screen.blit(t, t.get_rect(center=(c*CELL+CELL//2, r*CELL+CELL//2)))
            pygame.display.update()
        draw(); running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False
                if env.done: continue
                if e.type == pygame.MOUSEBUTTONDOWN and env.cur == X:
                    r, c = e.pos[1]//CELL, e.pos[0]//CELL; m = r*BOARD_SIZE + c
                    if env.board[r,c] == EMPTY:
                        env.step(m); draw();
                        if env.done: continue
                        env.step(bot.move(env, 0)); draw()
            pygame.time.wait(30)
        pygame.quit()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='az10x10.ckpt'); ap.add_argument('--train-iter', type=int, default=0); ap.add_argument('--overwrite', action='store_true');
    a = ap.parse_args(); learn = Learner()
    if os.path.exists(a.model) and not a.overwrite: learn.load(a.model)
    if a.overwrite or not os.path.exists(a.model):
        if a.train_iter == 0: a.train_iter = 100
    if a.train_iter: learn.learn(a.train_iter); learn.save(a.model)
    learn.play()

if __name__ == '__main__':
    main()