import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pygame
import board_games_fun as bfun      # game definitions
from board_graphical_interface import play_with_strategy
import matplotlib.pyplot as plt

# Initialise Pygame (needed for the graphical interface)
pygame.init()

# -------------------------------------------------------------------
# Utility: state -> tensor
# -------------------------------------------------------------------
def state_to_tensor(state):
    return np.stack([state == 1, state == 2], axis=-1).astype(np.float32)
bfun.Tictac_general.state_to_tensor = staticmethod(state_to_tensor)

# -------------------------------------------------------------------
# Potential‐based shaping (optional)
# -------------------------------------------------------------------
def potential(state, player):
    A = state; R, C = A.shape; best = 0
    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
        for r in range(R):
            for c in range(C):
                cnt = 0; rr, cc = r, c
                while 0 <= rr < R and 0 <= cc < C and A[rr,cc] == player:
                    cnt += 1; rr += dr; cc += dc
                best = max(best, cnt)
    return best / 5.0

# -------------------------------------------------------------------
# MCTS Node and Strategy_MCTS
# -------------------------------------------------------------------
class MCTSNode:
    def __init__(self, state, player, parent=None, action_idx=None):
        self.state, self.player = state, player
        self.parent, self.action_idx = parent, action_idx
        self.children = []
        self.N = self.W = self.Q = self.P = 0.0
    def expand(self, game):
        acts = game.actions(self.state, self.player)
        opp = 3 - self.player
        for i,a in enumerate(acts):
            s2,_ = game.next_state_and_reward(self.player, self.state, a)
            self.children.append(MCTSNode(s2, opp, parent=self, action_idx=i))
    def is_leaf(self):
        return not self.children

class Strategy_MCTS:
    def __init__(self, game, model,
                 n_simulations=200, c_puct=1.0, temp=0.0, max_depth=16):
        self.game, self.model = game, model
        self.n_sim, self.c, self.temp = n_simulations, c_puct, temp
        self.max_depth = max_depth
        self.board_size = game.num_of_rows * game.num_of_columns

    def _predict(self, state):
        x = state_to_tensor(state)[None]
        pi, v = self.model.predict(x, verbose=0)
        return pi.ravel(), float(v[0,0])

    def select(self, node):
        best, best_val = None, -1e9
        for c in node.children:
            u = self.c * c.P * math.sqrt(node.N)/(1+c.N)
            val = c.Q + u
            if val > best_val:
                best_val, best = val, c
        return best

    def simulate(self, root_state, root_player):
        root = MCTSNode(root_state, root_player)
        pi, v = self._predict(root_state)
        root.expand(self.game)
        for c in root.children:
            c.P = pi[c.action_idx]
        root.N, root.W, root.Q = 1, (v if root.player==1 else -v), (v if root.player==1 else -v)

        for _ in range(self.n_sim - 1):
            node, depth = root, 0
            while not node.is_leaf() and depth < self.max_depth:
                node = self.select(node); depth += 1
            if node.is_leaf() and depth < self.max_depth:
                pi2, v2 = self._predict(node.state)
                node.expand(self.game)
                for c in node.children:
                    c.P = pi2[c.action_idx]
                value = v2 if node.player==1 else -v2
            else:
                _, v2 = self._predict(node.state)
                value = v2 if node.player==1 else -v2
            cur, to_prop = node, value
            while cur:
                cur.N += 1; cur.W += to_prop; cur.Q = cur.W/cur.N
                to_prop = -to_prop; cur = cur.parent

        visits = np.array([c.N for c in root.children], dtype=float)
        policy = np.zeros(self.board_size, dtype=float)
        if self.temp == 0:
            idx = np.argmax(visits)
        else:
            visits = visits**(1/self.temp)
            visits /= visits.sum()
            idx = np.random.choice(len(visits), p=visits)
        move = self.game.actions(root_state, root_player)[idx]
        flat = move[0]*self.game.num_of_columns + move[1] if isinstance(move,tuple) else move
        policy[flat] = 1.0
        return idx, policy

    def choose_action(self, state, player):
        acts = self.game.actions(state, player)
        # immediate win
        for i,a in enumerate(acts):
            s2,_ = self.game.next_state_and_reward(player, state, a)
            if self.game.end_of_game((player==1)-(player==2),0,s2,None):
                p = np.zeros(self.board_size); flat=(a[0]*self.game.num_of_columns+a[1]) if isinstance(a,tuple) else a
                p[flat]=1.0; return i,p
        # block opponent
        opp = 3-player
        for i,a in enumerate(acts):
            s2,_ = self.game.next_state_and_reward(player, state, a)
            for b in self.game.actions(s2, opp):
                s3,_ = self.game.next_state_and_reward(opp, s2, b)
                if self.game.end_of_game((opp==1)-(opp==2),0,s3,None):
                    p = np.zeros(self.board_size); flat=(a[0]*self.game.num_of_columns+a[1]) if isinstance(a,tuple) else a
                    p[flat]=1.0; return i,p
        return self.simulate(state, player)

# -------------------------------------------------------------------
# Neural network (AlphaZero head)
# -------------------------------------------------------------------
def build_alpha_zero_net(rows, cols, n_filters=64, n_res_blocks=8):
    inp = layers.Input((rows, cols, 2))
    x = layers.Conv2D(n_filters,3,padding='same',activation='relu')(inp)
    for _ in range(n_res_blocks):
        res = x
        x = layers.Conv2D(n_filters,3,padding='same',activation='relu')(x)
        x = layers.Conv2D(n_filters,3,padding='same')(x)
        x = layers.Add()([res, x]); x = layers.Activation('relu')(x)
    p = layers.Conv2D(2,1,activation='relu')(x)
    p = layers.Flatten()(p)
    p = layers.Dense(rows*cols,activation='softmax',name='pi')(p)
    v = layers.Conv2D(1,1,activation='relu')(x)
    v = layers.Flatten()(v)
    v = layers.Dense(64,activation='relu')(v)
    v = layers.Dense(1,activation='tanh',name='v')(v)
    model = models.Model(inp, [p, v])
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss={'pi':'categorical_crossentropy','v':'mse'},
        loss_weights={'pi':1,'v':1}
    )
    return model

# -------------------------------------------------------------------
# Self-play episode with shaping
# -------------------------------------------------------------------
def self_play_episode(game, model, sims_per_move, shape_coef=0.1):
    mcts = Strategy_MCTS(game, model, n_simulations=sims_per_move, c_puct=1.0, temp=0.0)
    state = game.initial_state()
    player = 1
    trajectory = []
    phi = {1:potential(state,1), 2:potential(state,2)}

    while True:
        idx, policy = mcts.choose_action(state, player)
        trajectory.append((state, policy, player))
        old = phi[player]
        move = game.actions(state, player)[idx]
        state, reward = game.next_state_and_reward(player, state, move)
        phi[player] = potential(state, player)
        shaped = reward + shape_coef*(phi[player]-old)
        if game.end_of_game(reward,0,state,None):
            z = shaped
            break
        player = 3-player

    return [(s, p, z if pl==1 else -z) for (s,p,pl) in trajectory]

# -------------------------------------------------------------------
# Training loop with evaluation each iteration
# -------------------------------------------------------------------
def train_alpha_zero(game, model,
                     n_iterations=5,
                     episodes_per_iter=10,
                     sims_per_move=100):
    history_pi, history_v = [], []
    for it in range(1, n_iterations+1):
        print(f"\n=== Iteration {it}/{n_iterations} ===")
        # 1) Self-play data
        X, P, Z = [], [], []
        for _ in range(episodes_per_iter):
            for s,p,z in self_play_episode(game, model, sims_per_move):
                X.append(state_to_tensor(s))
                P.append(p)
                Z.append(z)
        X = np.stack(X); P = np.stack(P); Z = np.array(Z)

        # 2) Train
        hist = model.fit(
            X, {'pi':P, 'v':Z},
            batch_size=64, epochs=1, verbose=0
        ).history
        history_pi.append(hist.get('pi_loss', hist.get('loss')))
        history_v.append(hist.get('v_loss', 0))

        # 3) Save weights
        model.save_weights(f"az_it{it}.weights.h5")

        # 4) Evaluate Bot vs Bot
        hard = Strategy_MCTS(game, model,
                             n_simulations=sims_per_move,
                             c_puct=1.0, temp=0.0)
        wins = {1:0,2:0,'draw':0}
        for _ in range(10):  # play 10 games
            s0 = game.initial_state(); pl=1
            while True:
                idx,_ = hard.choose_action(s0,pl)
                s0, r = game.next_state_and_reward(pl, s0, game.actions(s0,pl)[idx])
                if game.end_of_game(r,0,s0,None):
                    if r==1: wins[1]+=1
                    elif r==-1: wins[2]+=1
                    else: wins['draw']+=1
                    break
                pl = 3-pl
        print(f" After iter {it}: Bot1 wins {wins[1]}, Bot2 wins {wins[2]}, Draws {wins['draw']}")

    # Plot learning curves
    plt.plot(range(1,n_iterations+1), history_pi, label='pi loss')
    plt.plot(range(1,n_iterations+1), history_v, label='v loss')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend()
    plt.show()

# -------------------------------------------------------------------
# Difficulty presets and factory
# -------------------------------------------------------------------
DIFFICULTIES = {
    "easy":   {"sims": 20,  "c_puct": 2.5, "temp": 1.0},
    "medium": {"sims": 100, "c_puct": 1.5, "temp": 0.5},
    "hard":   {"sims": 500, "c_puct": 1.0, "temp": 0.0},
}
def make_strategy(game, model, level):
    cfg = DIFFICULTIES[level]
    return Strategy_MCTS(game, model,
                        n_simulations=cfg["sims"],
                        c_puct=cfg["c_puct"],
                        temp=cfg["temp"])

# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
if __name__ == '__main__':
    game = bfun.Tictac_general(10, 10, 5, False)
    rows, cols = game.num_of_rows, game.num_of_columns

    model = build_alpha_zero_net(rows, cols)

    # Train from scratch or continue
    print("Starting AlphaZero training...")
    train_alpha_zero(game, model,
                     n_iterations=5,
                     episodes_per_iter=10,
                     sims_per_move=100)
    model.save_weights('az.weights.h5')

    # Finally, human plays
    lvl = input("\nChoose level [easy/medium/hard]: ").strip().lower()
    if lvl not in DIFFICULTIES:
        lvl = "medium"
    strat = make_strategy(game, model, lvl)
    print("You are X (player 1). Good luck!")
    play_with_strategy(game, strategy=strat, str_player=2)
