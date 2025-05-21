# Filename: A0.py
"""AlphaZero‑style Tic‑Tac‑Toe / Gomoku agent (CPU‑first)
--------------------------------------------------------
• trains from scratch or resumes from az.weights.h5
• lets you watch self‑play games during training or after
• three human‑play difficulty presets

Updates 2025‑05‑20
------------------
* **Fast batch inference** – one NN call for all children
* **Safer plotting** – headless backend + save PNG instead of plt.show()
* **Light test settings** – 2 iter × 3 games × 50 sims (change as you like)
"""

import os, math, time
from typing import Tuple, List, Dict

# ── Silence TensorFlow chatty logs ───────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide INFO/WARN
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # disable oneDNN banners

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# headless backend → avoids PyCharm tostring_rgb bug
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygame

import board_games_fun as bfun
from board_graphical_interface import play_with_strategy

pygame.init()

# ----------------------------------------------------------------------
# GPU / CPU info (just to show the user)
# ----------------------------------------------------------------------
_gpus = tf.config.list_physical_devices("GPU")
print("✅ GPUs:" if _gpus else "⚠️  No GPU – CPU only.", _gpus)

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------

def state_to_tensor(state: np.ndarray) -> np.ndarray:
    """(rows,cols) int array → (rows,cols,2) float32 tensor."""
    return np.stack([state == 1, state == 2], axis=-1).astype(np.float32)

bfun.Tictac_general.state_to_tensor = staticmethod(state_to_tensor)

# ----------------------------------------------------------------------
# MCTS
# ----------------------------------------------------------------------
class MCTSNode:
    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state; self.parent = parent; self.action_from_parent = action_from_parent
        self.children: List["MCTSNode"] = []
        self.N = 0; self.W = 0.; self.Q = 0.; self.P = 1.

    def expand(self, game, player):
        self.children = []
        for idx, a in enumerate(game.actions(self.state, player)):
            s2, _ = game.next_state_and_reward(player, self.state, a)
            self.children.append(MCTSNode(s2, self, idx))

    def is_leaf(self):
        return not self.children


class Strategy_MCTS:
    def __init__(self, game, n_simulations=100, c_puct=1.0, model=None, max_depth=16):
        self.game, self.n_sim, self.c, self.model, self.max_depth = game, n_simulations, c_puct, model, max_depth

    @staticmethod
    def _batch_infer(model: tf.keras.Model, states: np.ndarray):
        pol, val = model(states, training=False); return pol.numpy(), val.numpy().ravel()

    def select(self, node):
        return max(node.children,
                   key=lambda n: n.Q + self.c * n.P * math.sqrt(node.N) / (1 + n.N))

    def expand_and_evaluate(self, node, player):
        node.expand(self.game, player)
        batch = np.stack([state_to_tensor(ch.state) for ch in node.children])
        policies, values = self._batch_infer(self.model, batch)
        for ch, pol in zip(node.children, policies):
            ch.P = pol[ch.action_from_parent]
        return float(values.mean())

    def simulate(self, state, player):
        root = MCTSNode(state)
        val = self.expand_and_evaluate(root, player)
        root.N = 1; root.W = val if player == 1 else -val; root.Q = root.W
        for _ in range(self.n_sim - 1):
            node, pl, depth = root, player, 0
            while not node.is_leaf() and depth < self.max_depth:
                node = self.select(node); pl = 3 - pl; depth += 1
            value = self.expand_and_evaluate(node, pl) if depth < self.max_depth else -node.Q
            cur = node
            while cur:
                cur.N += 1; cur.W += value if pl == 1 else -value; cur.Q = cur.W / cur.N; cur = cur.parent
        counts = np.zeros(self.game.num_of_rows * self.game.num_of_columns)
        for ch in root.children:
            counts[ch.action_from_parent] = ch.N
        return int(np.argmax(counts)), counts / counts.sum()

    def choose_action(self, state, player):
        """Return (best_action_idx, policy_vector) as expected by the GUI."""
        idx, policy = self.simulate(state, player)
        return idx, policy

# ----------------------------------------------------------------------
# CNN
# ----------------------------------------------------------------------

def build_alpha_zero_net(rows, cols, n_filters=32, n_res_blocks=2):
    inp = layers.Input((rows, cols, 2))
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu")(inp)
    for _ in range(n_res_blocks):
        res = x
        x = layers.Conv2D(n_filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(n_filters, 3, padding="same")(x)
        x = layers.Add()([res, x]); x = layers.Activation("relu")(x)
    p = layers.Conv2D(2, 1, activation="relu")(x); p = layers.Flatten()(p)
    p = layers.Dense(rows * cols, activation="softmax", name="pi")(p)
    v = layers.Conv2D(1, 1, activation="relu")(x); v = layers.Flatten()(v)
    v = layers.Dense(64, activation="relu")(v); v = layers.Dense(1, activation="tanh", name="v")(v)
    model = models.Model(inp, [p, v])
    model.compile(optimizers.Adam(1e-3), {"pi": "categorical_crossentropy", "v": "mse"}, {"pi": 1, "v": 1})
    return model

# ----------------------------------------------------------------------
# Self‑play & training
# ----------------------------------------------------------------------

def self_play_episode(game, model, sims_per_move):
    mcts = Strategy_MCTS(game, sims_per_move, model=model)
    state, player = game.initial_state(), 1
    traj = []
    while True:
        idx, pol = mcts.simulate(state, player); traj.append((state, pol, player))
        state, reward = game.next_state_and_reward(player, state, game.actions(state, player)[idx])
        if game.end_of_game(reward, 0, state, None):
            return [(s, p, reward if pl == 1 else -reward) for (s, p, pl) in traj]
        player = 3 - player


def train_alpha_zero(
            game,
            model,
            n_iterations=10,        # deeper training
            episodes_per_iter=20,   # more self‑play games per iteration
            sims_per_move=100       # stronger search during training
        ):
    pi_loss, v_loss = [], []
    for it in range(1, n_iterations + 1):
        print(f"Iteration {it}/{n_iterations}"); t0 = time.time()
        X, P, Z = [], [], []
        for _ in range(episodes_per_iter):
            for s, p, z in self_play_episode(game, model, sims_per_move):
                X.append(state_to_tensor(s)); P.append(p); Z.append(z)
        X, P, Z = np.stack(X), np.stack(P), np.array(Z)
        h = model.fit(X, {"pi": P, "v": Z}, batch_size=32, epochs=1, verbose=0).history
        pi_loss.append(h["pi_loss"][-1]); v_loss.append(h["v_loss"][-1])
        print(f"  ↳ {len(Z)} moves in {time.time()-t0:.1f}s")
    plt.plot(pi_loss, label="policy loss"); plt.plot(v_loss, label="value loss"); plt.xlabel("iter"); plt.legend()
    plt.tight_layout(); plt.savefig("training_losses.png"); plt.close()

# ----------------------------------------------------------------------
# Difficulty presets
# ----------------------------------------------------------------------
DIFFICULTIES = {"easy": {"depth": 3, "sims": 30}, "medium": {"depth": 8, "sims": 60}, "hard": {"depth": 16, "sims": 120}}

def make_strategy(game, model, level):
    cfg = DIFFICULTIES[level]
    return Strategy_MCTS(game, cfg["sims"], model=model, max_depth=cfg["depth"])

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    game = bfun.Tictac_general(10, 10, 5, False)
    model = build_alpha_zero_net(game.num_of_rows, game.num_of_columns)
    if os.path.exists("az.weights.h5"):
        print("Loading existing weights …"); model.load_weights("az.weights.h5")
    else:
        print("No weights found → training from scratch …")
        train_alpha_zero(game, model)
        model.save_weights("az.weights.h5")

    level = input("Choose difficulty [easy/medium/hard]: ").strip().lower()
    level = level if level in DIFFICULTIES else "medium"
    play_with_strategy(game, make_strategy(game, model, level), str_player=1)
