import math
import copy
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# ---------- MCTS with neural network integration ----------
class MCTSNode:
    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = []
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = 1.0

    def expand(self, game_obj, player):
        actions = game_obj.actions(self.state, player)
        for idx, a in enumerate(actions):
            next_s, r = game_obj.next_state_and_reward(player, self.state, a)
            child = MCTSNode(next_s, parent=self, action_from_parent=idx)
            self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

class Strategy_MCTS:
    def __init__(self, game_obj, n_simulations=200, c_puct=1.0, model=None):
        self.game = game_obj
        self.n_sim = n_simulations
        self.c = c_puct
        self.model = model

    def select(self, node):
        best, best_val = None, -float('inf')
        for child in node.children:
            uct = child.Q + self.c * child.P * math.sqrt(node.N) / (1 + child.N)
            if uct > best_val:
                best_val, best = uct, child
        return best

    def expand_and_evaluate(self, node, player):
        node.expand(self.game, player)
        x = self.game.state_to_tensor(node.state)[None, ...]
        pi, v = self.model.predict(x, verbose=0)
        pi = pi.ravel()
        for child, p in zip(node.children, pi):
            child.P = p
        return v[0][0]

    def simulate(self, root_state, start_player):
        root = MCTSNode(root_state)
        v = self.expand_and_evaluate(root, start_player)
        root.N += 1
        root.W += v if start_player == 1 else -v
        root.Q = root.W / root.N

        for _ in range(self.n_sim - 1):
            node, player = root, start_player
            while not node.is_leaf():
                node = self.select(node)
                player = 3 - player
            v = self.expand_and_evaluate(node, player)
            cur = node
            while cur:
                cur.N += 1
                cur.W += v if player == 1 else -v
                cur.Q = cur.W / cur.N
                cur = cur.parent

        counts = [child.N for child in root.children]
        best_idx = int(np.argmax(counts))
        return best_idx, np.array(counts) / sum(counts)

    def choose_action(self, state, player):
        idx, _ = self.simulate(state, player)
        return idx

# ---------- AlphaZero Neural Network ----------
# build network for arbitrary rows x cols board
def build_alpha_zero_net(rows, cols, n_filters=64, n_res_blocks=3):
    inputs = layers.Input(shape=(rows, cols, 2), name='board')
    x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(inputs)
    for _ in range(n_res_blocks):
        skip = x
        x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(n_filters, 3, padding='same')(x)
        x = layers.Add()([skip, x])
        x = layers.Activation('relu')(x)
    p = layers.Conv2D(2, 1, activation='relu')(x)
    p = layers.Flatten()(p)
    p = layers.Dense(rows * cols, activation='softmax', name='pi')(p)
    v = layers.Conv2D(1, 1, activation='relu')(x)
    v = layers.Flatten()(v)
    v = layers.Dense(64, activation='relu')(v)
    v = layers.Dense(1, activation='tanh', name='v')(v)
    model = models.Model(inputs=inputs, outputs=[p, v], name='AlphaZeroNet')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={'pi': losses.CategoricalCrossentropy(), 'v': losses.MeanSquaredError()},
        loss_weights={'pi': 1.0, 'v': 1.0}
    )
    return model

# ---------- Self-play and Training Loop ----------
def self_play_episode(game, model, n_simulations):
    mcts = Strategy_MCTS(game, n_simulations=n_simulations, model=model)
    examples = []
    state = game.initial_state()
    player = 1
    while True:
        idx, policy = mcts.simulate(state, player)
        examples.append((state, policy, player))
        action_list = game.actions(state, player)
        action = action_list[idx]
        state, reward = game.next_state_and_reward(player, state, action)
        done = game.end_of_game(reward, 0, state, None)
        if done:
            z = reward
            break
        player = 3 - player
    return [(s, p, z if pl == 1 else -z) for (s, p, pl) in examples]


def train_alpha_zero(game, model, n_iterations=10, episodes_per_iter=20, sims_per_move=100):
    for it in range(1, n_iterations + 1):
        all_s, all_p, all_z = [], [], []
        print(f"Iteration {it}/{n_iterations}")
        for _ in range(episodes_per_iter):
            episode_data = self_play_episode(game, model, sims_per_move)
            for s, p, z in episode_data:
                all_s.append(game.state_to_tensor(s))
                all_p.append(p)
                all_z.append(z)
        X = np.array(all_s)
        P = np.array(all_p)
        Z = np.array(all_z)
        model.fit(X, {'pi': P, 'v': Z}, batch_size=64, epochs=1)
        model.save_weights(f"az_weights_iter{it}.h5")

# ---------- Utilities for generalized board ----------
def state_to_tensor(self, state):
    # two channels: X=1, O=2
    return np.stack([state == 1, state == 2], axis=-1).astype(np.float32)

# monkey-patch into Tictac_general
import board_games_fun as bfun
bfun.Tictac_general.state_to_tensor = state_to_tensor

# ---------- Main Execution ----------
if __name__ == "__main__":
    from board_graphical_interface import play_with_strategy
    import board_games_fun as bfun

    # generalized TicTacToe: 10x10, win in 5
    game = bfun.Tictac_general(10, 10, 5, False)
    rows, cols = game.num_of_rows, game.num_of_columns

    # Build or load model
    model = build_alpha_zero_net(rows, cols)
    weights_file = 'az_weights_latest.h5'

    if os.path.exists(weights_file):
        print("Loading existing AlphaZero model weights...")
        model.load_weights(weights_file)
    else:
        print("No saved model found. Training AlphaZero from scratch...")

    # Always train self-play
    train_alpha_zero(
        game,
        model,
        n_iterations=5,
        episodes_per_iter=20,
        sims_per_move=200
    )
    model.save_weights(weights_file)

    # Start human-vs-AI game
    play_with_strategy(
        game,
        strategy=lambda state, player: Strategy_MCTS(
            game_obj=game,
            n_simulations=200,
            model=model
        ).choose_action(state, player),
        human_player=2
    )
