# Filename: A0.py

import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pygame
import board_games_fun as bfun  # game definitions
from board_graphical_interface import play_with_strategy
import matplotlib.pyplot as plt

# Initialise Pygame (needed for the graphical interface)
pygame.init()

# -----------------------------------------------------------------------------
# Utility: state -> tensor ------------------------------------------------------
# -----------------------------------------------------------------------------

def state_to_tensor(state):
    """Convert a 2‑D board state into a (rows, cols, 2) float tensor.

    Channel 0 is the mask of player‑1 stones (value 1.0 where X stands),
    Channel 1 is the mask of player‑2 stones (value 1.0 where O stands).
    """
    return np.stack([state == 1, state == 2], axis=-1).astype(np.float32)

# Attach the helper to the game class so that other modules can use it ---------
bfun.Tictac_general.state_to_tensor = staticmethod(state_to_tensor)

# -----------------------------------------------------------------------------
# Monte‑Carlo Tree Search -------------------------------------------------------
# -----------------------------------------------------------------------------

class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent  # index in parent's action list
        self.children = []  # list[MCTSNode]
        # Book‑keeping ----------------------------------------------------------
        self.N = 0           # visit count
        self.W = 0.0        # total value of this node (from the root's P.O.V.)
        self.Q = 0.0        # mean value = W / N
        self.P = 1.0        # prior from the policy head

    # ---------------------------------------------------------------------
    # Expansion -----------------------------------------------------------
    # ---------------------------------------------------------------------

    def expand(self, game_obj, player):
        """Create child nodes for every legal action from *state*."""
        self.children = []
        actions = game_obj.actions(self.state, player)
        for idx, a in enumerate(actions):
            next_state, _ = game_obj.next_state_and_reward(player, self.state, a)
            self.children.append(MCTSNode(next_state, parent=self, action_from_parent=idx))

    def is_leaf(self):
        return not self.children

# -----------------------------------------------------------------------------
# Strategy wrapper that couples MCTS with the neural network -------------------
# -----------------------------------------------------------------------------

class Strategy_MCTS:
    """AlphaZero‑style Monte‑Carlo Tree Search with a depth limit."""

    def __init__(self, game_obj, n_simulations=100, c_puct=1.0, model=None, max_depth=16):
        self.game = game_obj
        self.n_sim = n_simulations
        self.c = c_puct
        self.model = model
        self.max_depth = max_depth

    # ---------------------------------------------------------------------
    # Selection -----------------------------------------------------------
    # ---------------------------------------------------------------------

    def select(self, node):
        """Pick the child with the highest PUCT score."""
        return max(
            node.children,
            key=lambda n: n.Q + self.c * n.P * math.sqrt(node.N) / (1 + n.N)
        )

    # ---------------------------------------------------------------------
    # Expansion + evaluation ---------------------------------------------
    # ---------------------------------------------------------------------

    def expand_and_evaluate(self, node, player):
        node.expand(self.game, player)
        x = state_to_tensor(node.state)[None]           # (1, rows, cols, 2)
        policy_full, v = self.model.predict(x, verbose=0)
        policy_full = policy_full.ravel()
        for child in node.children:
            child.P = policy_full[child.action_from_parent]
        return v[0][0]

    # ---------------------------------------------------------------------
    # Simulate one full playout ------------------------------------------
    # ---------------------------------------------------------------------

    def simulate(self, state, player):
        root = MCTSNode(state)
        value = self.expand_and_evaluate(root, player)
        root.N = 1
        root.W = value if player == 1 else -value
        root.Q = root.W

        # -- run n_sim‑1 additional simulations ---------------------------
        for _ in range(self.n_sim - 1):
            node, pl, depth = root, player, 0
            # 1) SELECTION -------------------------------------------------
            while not node.is_leaf() and depth < self.max_depth:
                node = self.select(node)
                pl = 3 - pl
                depth += 1
            # 2) EXPANSION + EVAL or leaf eval ----------------------------
            if depth < self.max_depth:
                value = self.expand_and_evaluate(node, pl)
            else:
                x = state_to_tensor(node.state)[None]
                _, v = self.model.predict(x, verbose=0)
                value = v[0][0]
            # 3) BACK‑PROP -------------------------------------------------
            cur = node
            while cur:
                cur.N += 1
                cur.W += value if pl == 1 else -value
                cur.Q = cur.W / cur.N
                cur = cur.parent

        # -----------------------------------------------------------------
        # Turn visit counts into a policy vector ---------------------------
        total = self.game.num_of_rows * self.game.num_of_columns
        counts = np.zeros(total, dtype=float)
        for child in root.children:
            counts[child.action_from_parent] = child.N
        policy = counts / counts.sum()
        best_idx = int(np.argmax(counts))
        return best_idx, policy

    # ---------------------------------------------------------------------
    # Public helper -------------------------------------------------------
    # ---------------------------------------------------------------------

    def choose_action(self, state, player):
        """Return (action_index, policy_vector)."""
        idx, policy = self.simulate(state, player)
        return idx, policy

# -----------------------------------------------------------------------------
# Neural network (AlphaZero head) ---------------------------------------------
# -----------------------------------------------------------------------------

def build_alpha_zero_net(rows, cols, n_filters=32, n_res_blocks=2):
    inputs = layers.Input((rows, cols, 2))
    x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(inputs)
    for _ in range(n_res_blocks):
        res = x
        x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(n_filters, 3, padding='same')(x)
        x = layers.Add()([res, x])
        x = layers.Activation('relu')(x)
    # Policy head ---------------------------------------------------------
    p = layers.Conv2D(2, 1, activation='relu')(x)
    p = layers.Flatten()(p)
    p = layers.Dense(rows * cols, activation='softmax', name='pi')(p)
    # Value head ----------------------------------------------------------
    v = layers.Conv2D(1, 1, activation='relu')(x)
    v = layers.Flatten()(v)
    v = layers.Dense(64, activation='relu')(v)
    v = layers.Dense(1, activation='tanh', name='v')(v)

    model = models.Model(inputs, [p, v])
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss={'pi': 'categorical_crossentropy', 'v': 'mse'},
        loss_weights={'pi': 1, 'v': 1}
    )
    return model

# -----------------------------------------------------------------------------
# Self‑play episode ------------------------------------------------------------
# -----------------------------------------------------------------------------

def self_play_episode(game, model, sims_per_move):
    mcts = Strategy_MCTS(game, n_simulations=sims_per_move, model=model, max_depth=16)
    state = game.initial_state()
    player = 1
    trajectory = []
    while True:
        idx, policy = mcts.simulate(state, player)
        trajectory.append((state, policy, player))
        state, reward = game.next_state_and_reward(
            player, state, game.actions(state, player)[idx])
        if game.end_of_game(reward, 0, state, None):
            z = reward
            break
        player = 3 - player
    # Map the final reward to each player’s perspective -------------------
    return [(s, p, z if pl == 1 else -z) for (s, p, pl) in trajectory]

# -----------------------------------------------------------------------------
# Training loop ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def train_alpha_zero(game, model,
                     n_iterations=3,
                     episodes_per_iter=5,
                     sims_per_move=50,
                     visualize_every=0):
    history_pi, history_v = [], []
    for it in range(1, n_iterations + 1):
        print(f"Iteration {it}/{n_iterations}")
        X, P, Z = [], [], []
        for _ in range(episodes_per_iter):
            for s, p, z in self_play_episode(game, model, sims_per_move):
                X.append(state_to_tensor(s))
                P.append(p)
                Z.append(z)
        X = np.stack(X)
        P = np.stack(P)
        Z = np.array(Z)
        hist = model.fit(
            X, {'pi': P, 'v': Z},
            batch_size=32, epochs=1, verbose=0
        ).history
        history_pi.append(hist.get('pi_loss', hist.get('loss')))
        history_v.append(hist.get('v_loss', 0))

        # Save intermediate weights so you can resume later --------------
        model.save_weights(f"az_it{it}.weights.h5")

        # Optional: watch the AI play against itself ----------------------
        if visualize_every and it % visualize_every == 0:
            ai_strategy = Strategy_MCTS(
                game, n_simulations=sims_per_move, model=model, max_depth=16)
            play_with_strategy(game, strategy=ai_strategy, str_player=0)

    # Plot learning curves ------------------------------------------------
    plt.plot(range(1, n_iterations + 1), history_pi, label='pi loss')
    plt.plot(range(1, n_iterations + 1), history_v, label='v loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# -----------------------------------------------------------------------------
# Difficulty presets -----------------------------------------------------------
# -----------------------------------------------------------------------------

DIFFICULTIES = {
    "easy":   {"depth": 3,  "sims": 30},
    "medium": {"depth": 8,  "sims": 60},
    "hard":   {"depth": 16, "sims": 120},
}

def make_strategy(game, model, level):
    cfg = DIFFICULTIES[level]
    return Strategy_MCTS(
        game, n_simulations=cfg["sims"], model=model, max_depth=cfg["depth"])

# -----------------------------------------------------------------------------
# Main entry point -------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Configure the board (rows, cols, in-a-row, wrap-around)
    game = bfun.Tictac_general(10, 10, 5, False)
    rows, cols = game.num_of_rows, game.num_of_columns

    # Build network ------------------------------------------------------
    model = build_alpha_zero_net(rows, cols)

    # Either load existing weights or train from scratch -----------------
    if os.path.exists('az.weights.h5'):
        print("Loading existing weights…")
        model.load_weights('az.weights.h5')
    else:
        print("No weights found → training from scratch …")
        train_alpha_zero(
            game, model,
            n_iterations=5,
            episodes_per_iter=10,
            sims_per_move=100,
            visualize_every=0
        )
        model.save_weights('az.weights.h5')

    # -------------------------------------------------------------------
    # Play against the AI -----------------------------------------------
    # -------------------------------------------------------------------
    mode = input("Choose a mode [easy / medium / hard]: ").strip().lower()
    if mode not in DIFFICULTIES:
        print("Unknown mode, defaulting to medium …")
        mode = 'medium'

    ai_strategy = make_strategy(game, model, mode)
    # str_player = 2 → you are player 1 (X) and start the game
    play_with_strategy(game, ai_strategy, str_player=2)
