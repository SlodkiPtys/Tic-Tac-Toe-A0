import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pygame
import board_games_fun as bfun  # your game definitions
from board_graphical_interface import play_with_strategy
import matplotlib.pyplot as plt

pygame.init()

# -----------------------------------------------------------------------------
# Utility: state -> tensor ---------------------------------------------------
# -----------------------------------------------------------------------------

def state_to_tensor(state):
    return np.stack([state == 1, state == 2], axis=-1).astype(np.float32)

bfun.Tictac_general.state_to_tensor = staticmethod(state_to_tensor)

# -----------------------------------------------------------------------------
# Flat‐index helper & threat‐finder -------------------------------------------
# -----------------------------------------------------------------------------

def flat_index(move, cols):
    """Convert (r, c) into flattened index."""
    return move[0] * cols + move[1]

def find_threats(game, state, threat_player):
    """Return list of flat‐indices where threat_player would win immediately."""
    threats = []
    for a in game.actions(state, threat_player):
        _, reward = game.next_state_and_reward(threat_player, state, a)
        if reward == 1:
            threats.append(flat_index(a, game.num_of_columns))
    return threats

# -----------------------------------------------------------------------------
# Monte-Carlo Tree Search ----------------------------------------------------
# -----------------------------------------------------------------------------

class MCTSNode:
    def __init__(self, state, parent=None, action_from_parent=None, player_to_move=1):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.player_to_move = player_to_move
        self.children = []
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = 0.0

    def expand(self, game_obj):
        actions = game_obj.actions(self.state, self.player_to_move)
        next_player = 3 - self.player_to_move
        for a in actions:
            next_state, _ = game_obj.next_state_and_reward(self.player_to_move, self.state, a)
            flat = flat_index(a, game_obj.num_of_columns)
            child = MCTSNode(next_state,
                             parent=self,
                             action_from_parent=flat,
                             player_to_move=next_player)
            self.children.append(child)

    def is_leaf(self):
        return not self.children

class Strategy_MCTS:
    def __init__(self, game_obj, n_simulations=200, c_puct=1.0, model=None,
                 max_depth=16, dirichlet_eps=0.25, dirichlet_alpha=0.3, temperature=1.0):
        self.game = game_obj
        self.n_sim = n_simulations
        self.c = c_puct
        self.model = model
        self.max_depth = max_depth
        self.eps = dirichlet_eps
        self.alpha = dirichlet_alpha
        self.tau = temperature

    def select(self, node):
        return max(
            node.children,
            key=lambda n: n.Q + self.c * n.P * math.sqrt(node.N) / (1 + n.N)
        )

    def expand_and_evaluate(self, node, is_root=False):
        node.expand(self.game)
        x = state_to_tensor(node.state)[None]
        policy_full, v = self.model.predict(x, verbose=0)
        policy = policy_full.ravel()

        if is_root:
            noise = np.random.dirichlet([self.alpha] * policy.size)
            policy = (1 - self.eps) * policy + self.eps * noise

        for child in node.children:
            child.P = policy[child.action_from_parent]

        value = float(v[0][0])
        return value if node.player_to_move == 1 else -value

    def simulate(self, root_state, root_player):
        root = MCTSNode(root_state, player_to_move=root_player)
        value = self.expand_and_evaluate(root, is_root=True)
        root.N, root.W, root.Q = 1, value, value

        for _ in range(self.n_sim - 1):
            node = root
            depth = 0
            while not node.is_leaf() and depth < self.max_depth:
                node = self.select(node)
                depth += 1

            if node.is_leaf() and depth < self.max_depth:
                value = self.expand_and_evaluate(node)
            else:
                x = state_to_tensor(node.state)[None]
                _, v = self.model.predict(x, verbose=0)
                raw = float(v[0][0])
                value = raw if node.player_to_move == 1 else -raw

            cur = node
            while cur:
                cur.N += 1
                cur.W += value
                cur.Q = cur.W / cur.N
                value = -value
                cur = cur.parent

        counts = np.array([c.N for c in root.children], dtype=float)
        if self.tau > 0:
            probs = counts ** (1 / self.tau)
            probs /= probs.sum()
            idx = np.random.choice(len(counts), p=probs)
        else:
            idx = int(counts.argmax())

        policy = counts / counts.sum()
        return idx, policy

    def choose_action(self, state, player):
        return self.simulate(state, player)

# -----------------------------------------------------------------------------
# Neural net ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_alpha_zero_net(rows, cols, n_filters=64, n_res_blocks=10):
    inputs = layers.Input((rows, cols, 2))
    x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(inputs)
    for _ in range(n_res_blocks):
        res = x
        x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(n_filters, 3, padding='same')(x)
        x = layers.Add()([res, x])
        x = layers.Activation('relu')(x)
    p = layers.Conv2D(2, 1, activation='relu')(x)
    p = layers.Flatten()(p)
    p = layers.Dense(rows * cols, activation='softmax', name='pi')(p)
    v = layers.Conv2D(1, 1, activation='relu')(x)
    v = layers.Flatten()(v)
    v = layers.Dense(64, activation='relu')(v)
    v = layers.Dense(1, activation='tanh', name='v')(v)
    model = models.Model(inputs, [p, v])
    # double‐weight the value loss to push for wins
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss={'pi': 'categorical_crossentropy', 'v': 'mse'},
        loss_weights={'pi': 1, 'v': 2}
    )
    return model

# -----------------------------------------------------------------------------
# Self-play with defense bonus -----------------------------------------------
# -----------------------------------------------------------------------------

def self_play_episode(game, model, sims_per_move):
    mcts = Strategy_MCTS(game, n_simulations=sims_per_move, model=model,
                         max_depth=16, temperature=1.0)
    state = game.initial_state()
    player = 1
    trajectory = []
    last_threat = None

    while True:
        # Find one‐move threats by the opponent
        threats = find_threats(game, state, 3 - player)
        idx, policy = mcts.simulate(state, player)
        move = game.actions(state, player)[idx]
        flat = flat_index(move, game.num_of_columns)

        # If we block their threat, give a small bonus
        blocked = (flat in threats)
        bonus = 0.1 if blocked else 0.0

        trajectory.append((state, policy, player, bonus))
        state, reward = game.next_state_and_reward(player, state, move)

        if game.end_of_game(reward, 0, state, None):
            z = reward
            break

        player = 3 - player

    # Convert to training data (s, π, z), adding block‐bonus
    examples = []
    for (s, p, pl, bonus) in trajectory:
        z_pl = z if pl == 1 else -z
        examples.append((s, p, z_pl + bonus))
    return examples

# -----------------------------------------------------------------------------
# Training loop ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def train_alpha_zero(game, model,
                     n_iterations=20,
                     episodes_per_iter=50,
                     sims_per_move=200,
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
        X, P, Z = np.stack(X), np.stack(P), np.array(Z)

        hist = model.fit(
            X, {'pi': P, 'v': Z},
            batch_size=64, epochs=2, verbose=1
        ).history
        history_pi.append(hist['pi_loss'][0])
        history_v.append(hist['v_loss'][0])

        model.save_weights(f"az_it{it}.weights.h5")
        if visualize_every and it % visualize_every == 0:
            ai = Strategy_MCTS(game, sims_per_move, model, max_depth=16)
            play_with_strategy(game, ai, str_player=0)

    plt.plot(range(1, n_iterations+1), history_pi, label='pi loss')
    plt.plot(range(1, n_iterations+1), history_v, label='v loss')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend()
    plt.show()

# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    game = bfun.Tictac_general(10, 10, 5, False)
    rows, cols = game.num_of_rows, game.num_of_columns

    model = build_alpha_zero_net(rows, cols)

    if os.path.exists('az.weights.h5'):
        print("Loading existing weights…")
        model.load_weights('az.weights.h5')
    else:
        print("Training from scratch …")
        train_alpha_zero(game, model,
                         n_iterations=20,
                         episodes_per_iter=50,
                         sims_per_move=200,
                         visualize_every=0)
        model.save_weights('az.weights.h5')

    mode = input("Choose a mode [easy / medium / hard]: ").strip().lower()
    if mode not in ["easy", "medium", "hard"]:
        print("Unknown mode, defaulting to medium …")
        mode = 'medium'

    # Use pure MCTS here; if you want minimax fallback, you'd wrap it similarly
    strategy = Strategy_MCTS(
        game,
        n_simulations={'easy':30,'medium':60,'hard':120}[mode],
        model=model,
        max_depth={'easy':3,'medium':8,'hard':16}[mode]
    )
    play_with_strategy(game, strategy, str_player=2)
