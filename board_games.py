import numpy as np
import pickle
import time
import matplotlib.pylab as plt
import pdb
import board_games_fun as bfun
import board_graphical_interface as bgra


# Q-learning xo strategy using dictionary for Q-values in each known state:
class Strategy_Qdict:
    def __init__(self, game_obj):
        self.game_obj = game_obj  # game object - tic-tac-toe, generalized tic-tac, chess, ...
        self.Q_dict = {}  # dictionary - key: state, value: Q-table for al possible actions
        self.if_mixed = False  # if mixed strategy -> chosen from probability distribution
        self.distrib_dict = {}  # dictionary - key: state, value: cumulated sums of action probablities
        # to quick choose random action for mixed strategy

    # general methods:

    def choose_action(self, state, player):
        key = self.game_obj.state_key(state)  # making key for dictionary from state

        if self.if_mixed:
            if key in self.distrib_dict:
                cumdistr = np.cumsum(self.distrib_dict[key])
                x = np.random.random()
                index = len(cumdistr) - 1
                for i in range(len(cumdistr)):
                    if x < cumdistr[i]:
                        index = i
                        break
                return index, self.Q_dict[key][index]
            else:
                return None, 0
        else:
            if key in self.Q_dict:
                if player == 1:
                    action_no, value = np.argmax(self.Q_dict[key]), np.max(self.Q_dict[key])
                else:
                    action_no, value = np.argmin(self.Q_dict[key]), np.min(self.Q_dict[key])

                return action_no, value
            else:
                return None, 0

    def set_experience(self, state, action, reward, state_next):
        pass

    def to_file(self, filename):
        with open(filename, 'wb') as fp: pickle.dump(self.Q_dict, fp)
        print("strategy with " + str(len(self.Q_dict)) + " is stored in a file " + filename)
        with open(filename + ".txt", 'w') as fp: fp.write(str(self.Q_dict))

    def from_file(self, filename):
        with open(filename, 'rb') as fp: self.Q_dict = pickle.load(fp)

    # methods specific to the Q-learning dictionary method:

    def get_action_value(self, state, action_no):
        key = self.game_obj.state_key(state)  # making key for dictionary from state
        if key in self.Q_dict:
            Qtable = self.Q_dict[key]
            return Qtable[action_no]
        else:
            return 0

    def get_Qtable(self, state, player):
        key = self.game_obj.state_key(state)  # making key for dictionary from state
        if key in self.Q_dict:
            Qtable = self.Q_dict[key]
            return Qtable
        else:
            actions = self.game_obj.actions(state, player)
            return np.zeros(len(actions), dtype=float)

    def set_action_value(self, state, player, action_no, value):
        key = self.game_obj.state_key(state)  # making key for dictionary from state
        if (key in self.Q_dict) == False:
            actions = self.game_obj.actions(state, player)
            self.Q_dict[key] = np.zeros(len(actions), dtype=float)
            x = 1
        self.Q_dict[key][action_no] = value

    def make_epsilon_greedy(self, player, epsilon):
        for key, Qtab in self.Q_dict.items():
            distrib = np.zeros(len(Qtab), dtype=float)
            if player == 1:
                distrib[np.argmax(Qtab)] = 1.0
            else:
                distrib[np.argmin(Qtab)] = 1.0
            self.distrib_dict[key] = distrib
        self.if_mixed = True
        for key, distrib in self.distrib_dict.items():
            number_of_actions = len(distrib)
            distrib_new = np.zeros([len(distrib)], dtype=float)
            if (number_of_actions > 1):
                for i in range(number_of_actions):
                    distrib_new[i] = distrib[i] * (1 - epsilon) + epsilon / (number_of_actions - 1)
            self.distrib_dict[key] = distrib_new

    def make_pure(self):
        self.if_mixed = False


class Strategy_PolicyNeuro:
    def __init__(self, game_obj):
        self.game_obj = game_obj  # game object - tic-tac-toe, generalized tic-tac, chess, ...
        # ......................
        # ......................
        # ......................
        # ......................


class Strategy_VNeuro:
    def __init__(self, game_obj):
        self.game_obj = game_obj  # game object - tic-tac-toe, generalized tic-tac, chess, ...
        # ......................
        # ......................
        # ......................
        # ......................


# train function using Q table - more general approach: for each agent his opponent is
# treated as an envoronment. The Q-value update is delayed up to opponent's move due to
# the potentially random opponent strategy.
# inputs:
#    game_object - Tictactoe, Tictac_general, Connect4, Chess, etc ...
#    players_to_train - name of player for strategy training:
#        [1] for 'x', [2] for "o" or [1,2] for both players
#    strategy_x - strategy for player x - probability of each possible move in each state
#    strategy_o - strategy for player o
#    x is the player who starts game (in chess white pieces player)
#    choose_random - numbers of moves with puting x or o into random empty cell
#    (odd numbers {1,3,5,7,9} for x, even numbers {2,4,6,8} for o):
# output:??each possible move in each state)
def board_game_train_Q(game_object, players_to_train, strategy_x=None, strategy_o=None, number_of_games=1000,
                       choose_random=[]):
    # number_of_games = 2000   # e.g. number of episodes
    epsylon = 0.3  # exploration factor
    T = 3
    Tmin = 0.3
    if_softmax = False  # softmax (True) with T or epsilon-greedy (False) with epsilon random factor
    alpha = 0.5  # learning speed factor
    alpha_min = 0.01
    # alpha and epsylon can be changed in time
    gamma = 0.9  # discount factor (if < 1 near rewards are better than far, far punishments are better than near)
    lambda_ = 0  # fresh factor (if 0 then without eligibility traces)
    if_sarsa = 0  # 1 - SARSA
    # 0 - Q-learning
    if_Q_from_file = False
    log_to_file = False

    t1 = time.time()

    dT = (T - Tmin) / number_of_games  # change of temperature - softmax randomness parameter
    walpha = 1  # np.power(alpha_min/alpha,1/number_of_games)

    if strategy_x == None:
        strategy_x = Strategy_Qdict(game_object)
    if strategy_o == None:
        strategy_o = Strategy_Qdict(game_object)

    # W ramach uczenia do wykorzystania wraz z drzewami MCTS należy tablice Q zastąpić aproksymatorami
    # (lub aproksymatorem) funkcji użyteczności i funkcji strategii:
    # .............................................................
    if if_Q_from_file:
        strategy_x.from_file('Q_x.pkl')
        strategy_o.from_file('Q_o.pkl')
        print("Q-values loaded from files!")

    if log_to_file:
        f = open("board_game_train_Q.txt", "w")
        f.write("... start training by " + str(number_of_games) + " games\n")
        f.write("epsilon = " + str(epsylon) + " softmax = " + str(if_softmax) + " alpha = " + str(
            alpha) + " gamma = " + str(gamma) + " sarsa = " + str(if_sarsa) + "\n")

    for game_nr in range(number_of_games):  # episodes loop
        if (game_nr * 100) % number_of_games == 0: print("game = " + str(game_nr))
        if log_to_file:
            f.write("game = " + str(game_nr) + "\n")
        State = game_object.initial_state()  # initial state - empty board in tic-tac

        player = 1  # first movement by player 1(cross)

        if_end_of_loop = False  # the one more step is used for update the last action value
        if_end_of_game = False
        step_number = 0
        Reward_oppo = 0  # opponents reward (after opponent's action)

        ToUpdate1 = []
        ToUpdate2 = []

        while (if_end_of_loop == False):  # episode steps loop
            step_number += 1
            Reward = 0

            if player == 1:
                strategy = strategy_x
            else:
                strategy = strategy_o

            if not if_end_of_game:
                # Possible next states (2D arrays) for oponent (== current player actions) + rewards for each action
                actions = game_object.actions(State, player)

                num_of_actions = len(actions)
                # pdb.set_trace()         # traceing (c for continue)

                if player in players_to_train:
                    Q_state_actions = strategy.get_Qtable(State, player)
                    ind_Q_best, Q_best = strategy.choose_action(State, player)

                if step_number in choose_random:
                    action_nr = np.random.randint(num_of_actions)
                elif player not in players_to_train:  # using fixed strategy
                    action_nr, value = strategy.choose_action(State, player)
                    if action_nr == None:
                        action_nr = np.random.randint(num_of_actions)
                else:  # using learned strategy with exploration
                    if if_softmax:
                        distrib = bfun.softmax((1 / T) * Q_state_actions)
                        action_nr = bfun.choose_action(distrib)
                    elif (np.random.random() < epsylon) | (ind_Q_best == None):  # epsilon-greedy exploration
                        action_nr = np.random.randint(num_of_actions)  # next state random choose
                    else:
                        action_nr = ind_Q_best  # next state optimal-known choose

                NextState, Reward = game_object.next_state_and_reward(player, State, actions[action_nr])
                if log_to_file:
                    game_object.move_verification(State, actions, NextState, player, f)
                    f.close()
                    f = open("board_game_train_Q.txt", "a")
                    f.write(" gracz " + str(player) + " wykonal akcje " + str(action_nr) + " R = " + str(Reward) + "\n")
                    # f.write("w stanie:\n"+state_key+" gracz "+str(player)+" wykonal akcje "+str(action_nr)+" R = "+str(Reward)+"\n")

            if player in players_to_train:
                # Update information from previous player step:
                if step_number > 2:
                    if player == 1:
                        State_prev, action_nr_prev, Reward_prev = ToUpdate1
                        ToUpdate1 = []
                    else:
                        State_prev, action_nr_prev, Reward_prev = ToUpdate2
                        ToUpdate2 = []

                    if if_end_of_game:
                        Q_next = 0
                    else:
                        if if_sarsa:
                            Q_next = Q_state_actions[action_nr]
                        else:
                            Q_next = Q_best

                    Q_prev = strategy.get_action_value(State_prev, action_nr_prev)
                    value = Q_prev + alpha * (Reward_prev + Reward_oppo + gamma * Q_next - Q_prev)
                    strategy.set_action_value(State_prev, player, action_nr_prev, value)

                    if log_to_file:
                        f.write("update Q akcji " + str(action_nr_prev) + " w stanie:\n" + str(State_prev) + \
                                " R = " + str(Reward_oppo + Reward_prev) + " Q = " + str(value) + "\n")
                        f.write("Reward_prev = " + str(Reward_prev) + " Reward_oppo = " + str(Reward_oppo) + \
                                " Q_next = " + str(Q_next) + "\n")

                # Informations for future update (opponent move is unknown in this moment)
                if not if_end_of_game:
                    if player == 1:
                        ToUpdate1 = [State, action_nr, Reward]
                    else:
                        ToUpdate2 = [State, action_nr, Reward]
                    # pdb.set_trace()

            State = NextState  # move to next state
            player = 3 - player  # player changing
            Reward_oppo = Reward

            if if_end_of_game:
                if_end_of_loop = True  # if all updates are done

            if game_object.end_of_game(Reward, step_number, State, action_nr):  # win or draw
                if_end_of_game = True
                if log_to_file:
                    f.write("chyba koniec gry\n")

        # after the end of game:
        if len(ToUpdate1) > 0:
            State_prev, action_nr_prev, Reward_prev = ToUpdate1

            Q_prev = strategy_x.get_action_value(State_prev, action_nr_prev)
            value = Q_prev + alpha * (Reward_prev + Reward_oppo - Q_prev)
            strategy_x.set_action_value(State_prev, 1, action_nr_prev, value)
            if log_to_file:
                f.write("update Q akcji " + str(action_nr_prev) + " w stanie:\n" + str(State_prev) + " R = " + \
                        str(Reward_oppo + Reward_prev) + " Q = " + str(value) + "\n")
        if len(ToUpdate2) > 0:
            State_prev, action_nr_prev, Reward_prev = ToUpdate2

            Q_prev = strategy_o.get_action_value(State_prev, action_nr_prev)
            value = Q_prev + alpha * (Reward_prev + Reward_oppo - Q_prev)
            strategy_o.set_action_value(State_prev, 2, action_nr_prev, value)
            if log_to_file:
                f.write("update Q akcji " + str(action_nr_prev) + " w stanie:\n" + str(State_prev) + " R = " + \
                        str(Reward_oppo + Reward_prev) + " Q = " + str(value) + "\n")
        if log_to_file:
            f.write("zamkniecie epizodu " + str(game_nr) + "\n")
        T -= dT
        alpha *= walpha

    dt = time.time() - t1
    print("training finished after %.3f sec. (%.3f sec./1000 games)" % (dt, dt * 1000 / number_of_games))
    # print("number of states for x = " + str(len(strategy_x)) + " for o = "+ str(len(strategy_o)))
    if log_to_file:
        # f.write("training finished after %.3f sec. (%.3f sec./1000 games)\n" % (dt, dt*1000/number_of_games) )
        # f.write("number of states for x = " + str(len(strategy_x)) + " for o = "+ str(len(strategy_o))+"\n")
        f.close()

    strategy_x.to_file('Q_x.pkl')
    strategy_o.to_file('Q_o.pkl')
    # pdb.set_trace()

    return strategy_x, strategy_o


# Q-learning, but update for each player based on opponent's action in next step
# (not based on player actions after 2 steps as above)
def board_game_train_Q2(game_object, players_to_train, strategy_x=None, strategy_o=None, number_of_games=2000):
    epsylon = 0.3  # exploration factor
    T = 3
    Tmin = 0.3
    if_softmax = False  # softmax (True) with T or epsilon-greedy (False) with epsilon random factor
    alpha = 0.5  # learning speed factor
    alpha_min = 0.01
    # alpha and epsylon can be changed in time
    gamma = 0.9  # discount factor (if < 1 near rewards are better than far, far punishments are better than near)
    lambda_ = 0  # fresh factor (if 0 then without eligibility traces)
    if_sarsa = 0  # 1 - SARSA
    # 0 - Q-learning
    if_Q_from_file = False

    t1 = time.time()

    # numbers of moves with puting x or o into random empty cell (odd numbers {1,3,5,7,9} for x, even numbers {2,4,6,8} for o):
    choose_random = []

    dT = (T - Tmin) / number_of_games  # change of temperature - softmax randomness parameter
    walpha = np.power(alpha_min / alpha, 1 / number_of_games)

    if strategy_x == None:
        strategy_x = Strategy_Qdict(game_object)
    if strategy_o == None:
        strategy_o = Strategy_Qdict(game_object)

    if if_Q_from_file:
        try:
            strategy_x.from_file('Q_x.pkl')
            print("Q-values for x loaded from files!")
        except:
            pass
        try:
            strategy_o.from_file('Q_o.pkl')
            print("Q-values for o loaded from files!")
        except:
            pass
    else:
        pass
    print("... start training by " + str(number_of_games) + " games")

    for game_nr in range(number_of_games):  # episodes loop
        if (game_nr * 100) % number_of_games == 0: print("game = " + str(game_nr))
        State = game_object.initial_state()  # initial state - empty board in tic-tac

        player = 1  # first movement by player 1(cross)

        if_end_of_loop = False  # the one more step is used for update the last action value
        if_end_of_game = False
        step_number = 0

        while (if_end_of_loop == False):  # episode steps loop
            step_number += 1
            Reward = 0

            if not if_end_of_game:
                # Possible next states (2D arrays) for oponent (== current player actions) + rewards for each action
                actions = game_object.actions(State, player)

                num_of_actions = len(actions)

                if player == 1:
                    strategy = strategy_x
                else:
                    strategy = strategy_o

                if player in players_to_train:
                    Q_state_actions = strategy.get_Qtable(State, player)

                    ind_Q_best, Q_best = strategy.choose_action(State, player)

                if step_number in choose_random:
                    action_nr = np.random.randint(num_of_actions)
                elif player not in players_to_train:  # using fixed strategy

                    action_nr, value = strategy.choose_action(State, player)
                    if action_nr == None:
                        action_nr = np.random.randint(num_of_actions)
                else:  # using learned strategy with exploration
                    if if_softmax:
                        distrib = bfun.softmax((1 / T) * Q_state_actions)
                        action_nr = bfun.choose_action(distrib)
                    elif (np.random.random() < epsylon) | (ind_Q_best == None):  # epsilon-greedy exploration
                        action_nr = np.random.randint(num_of_actions)  # next state random choose
                    else:
                        action_nr = ind_Q_best  # next state optimal-known choose

                NextState, Reward = game_object.next_state_and_reward(player, State, actions[action_nr])

            # Informations for future update (opponent move is unknown in this moment)
            if player == 1:
                ToUpdate1 = [State, action_nr, Reward]
            else:
                ToUpdate2 = [State, action_nr, Reward]

            if (3 - player) in players_to_train:

                # Update information from previous player step:
                if step_number > 1:
                    if player == 1:
                        State_prev, action_nr_prev, Reward_prev = ToUpdate2
                        strategy_to_update = strategy_o
                    else:
                        State_prev, action_nr_prev, Reward_prev = ToUpdate1
                        strategy_to_update = strategy_x

                    if if_end_of_game:
                        Q_next = 0
                    else:
                        if if_sarsa:
                            Q_next = Q_state_actions[action_nr]
                        else:
                            Q_next = Q_best

                    Q_prev = strategy_to_update.get_action_value(State_prev, action_nr_prev)
                    value = Q_prev + alpha * (Reward_prev + gamma * Q_next - Q_prev)
                    strategy_to_update.set_action_value(State_prev, 3 - player, action_nr_prev, value)

            State = NextState  # move to next state
            player = 3 - player  # player changing

            if if_end_of_game:
                if_end_of_loop = True  # if all updates are done

            if game_object.end_of_game(Reward, step_number, State, action_nr):  # win or draw
                if_end_of_game = True

        T -= dT
        alpha *= walpha

    dt = time.time() - t1
    print("training finished after %.3f sec. (%.3f sec./1000 games)" % (dt, dt * 1000 / number_of_games))
    # print("number of states for x = " + str(len(strategy_x)) + " for o = "+ str(len(strategy_o)))
    strategy_x.to_file('Q_x.pkl')
    strategy_o.to_file('Q_o.pkl')
    return strategy_x, strategy_o


# Test of given game (game_object) and strategies for player x and o
# choose_random - numbers of moves with puting x or o into random empty cell
# (odd numbers {1,3,5,7,9} for x, even numbers {2,4,6,8} for o):
def board_game_test(game_object, strategy_x, strategy_o, number_of_games=100, choose_random=[]):
    num_win_x = 0
    num_win_o = 0
    num_draws = 0

    Games = []
    Rewards = []

    for game in range(number_of_games):  # episodes loop
        # print("game = " + str(game))
        State = game_object.initial_state()  # initial state - empty board in tictac
        player = 1  # first movement by player 1(cross)
        if_end = False
        step_number = 0
        States = []
        Actions = []

        States.append(State)

        while (if_end == False):  # episode steps loop
            step_number += 1

            actions = game_object.actions(State, player)

            if player == 1:
                strategy = strategy_x
            else:
                strategy = strategy_o

            action_nr, value = strategy.choose_action(State, player)
            if (action_nr == None) | (step_number in choose_random):
                action_nr = np.random.randint(len(actions))

            NextState, Reward = game_object.next_state_and_reward(player, State, actions[action_nr])

            State = NextState  # move to next state
            Actions.append(actions[action_nr])
            States.append(State)  # board for game description

            player = 3 - player  # player changing

            if game_object.end_of_game(Reward, step_number, State, action_nr):  # win or draw
                if_end = True
                if Reward == 1:
                    num_win_x += 1
                elif Reward == -1:
                    num_win_o += 1
                elif Reward == 0:
                    num_draws += 1
                Rewards.append(Reward)
        Games.append([States, Actions])
    return num_win_x, num_win_o, num_draws, Games, Rewards


def experiment_par_train():
    print("\nUCZENIE DWOCH STRATEGII JEDNOCZESNIE\n")
    game = bfun.Tictactoe()  # game class object
    game = bfun.Tictac_general(4, 4, 3, False)
    game = bfun.Tictac_general(10, 10, 4, False)
    # game = bfun.Tictac_general(5,5,4,False)
    # game = bfun.Connect4()
    # game = bfun.Chess("szachy_plansza_3x3.txt")
    # game = bfun.Chess("szachy_plansza_4x4.txt")
    # game = bfun.Chess("szachy_plansza_5x5.txt")
    # game = bfun.Chess("szachy_plansza_5x3_bez_kroli.txt")
    # game = bfun.Chess("szachy_plansza_5x10.txt")
    # game = bfun.Chess("szachy_plansza_standardowa.txt")
    # game = bfun.Chess("szachy_plansza_14x14.txt")

    strategy_x, strategy_o = board_game_train_Q2(game, players_to_train=[1, 2], number_of_games=10)

    bgra.play_with_strategy(game_object=game, strategy=strategy_o, str_player=2)

    print("test stategii uczonych jednocześniewl,:")
    num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, strategy_x, strategy_o, choose_random=[])
    print("liczby wygranych: x = " + str(num_win_x) + ", o = " + str(num_win_o) + ", l.remisów = " + str(num_draws))
    game.print_test_to_file("gry_wyuczonych_strategii.txt", num_win_x, num_win_o, num_draws, Games, Rewards)

    print("test stategii x na częściowo losowej o:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    for i in range(10):
        epsilon = i / 10
        t.append(epsilon)

        strategy_o.make_epsilon_greedy(player=2, epsilon=epsilon)
        num_win_x, num_win_o, num_draws, Games, Rewards = \
            board_game_test(game, strategy_x, strategy_o)
        game.print_test_to_file("gry_x_vs_losowe_o_epsilon" + str(epsilon) + ".txt", \
                                num_win_x, num_win_o, num_draws, Games, Rewards)

        nwin_x.append(num_win_x)
        nwin_o.append(num_win_o)
        ndraws.append(num_draws)
    strategy_o.make_pure()
    plt.plot(t, nwin_x, "x", t, nwin_o, "o", t, ndraws, "-")
    plt.title("tictac test results with Nash x strategy and partially random o strategy")
    plt.xlabel("randomness of o strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of x wins", "num.of o wins", "num of draws"])
    plt.savefig("test_Nash_x_strategy_random_o_strategy.png")
    fig1 = plt
    plt.show()

    print("test stategii o na częściowo losowej x:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    for i in range(10):
        epsilon = i / 10
        t.append(epsilon)
        strategy_x.make_epsilon_greedy(player=1, epsilon=epsilon)
        num_win_x, num_win_o, num_draws, Games, Rewards = \
            board_game_test(game, strategy_x, strategy_o)
        game.print_test_to_file("gry_losowe_x_epsilon" + str(epsilon) + "_vs_o.txt", \
                                num_win_x, num_win_o, num_draws, Games, Rewards)
        nwin_x.append(num_win_x)
        nwin_o.append(num_win_o)
        ndraws.append(num_draws)
    strategy_x.make_pure()
    plt.plot(t, nwin_x, "x", t, nwin_o, "o", t, ndraws, "-")
    plt.title("tictac test results with Nash o strategy and partially random x strategy")
    plt.xlabel("randomness of x strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of x wins", "num.of o wins", "num of draws"])
    plt.savefig("test_Nash_o_strategy_random_x_strategy.png")
    plt.show()
    fig2 = plt

    # play with x (white in chess) strategy:
    # bgra.play_with_strategy(game_object = game, strategy = strategy_x, str_player=1)

    # play with o (black in chess) strategy:
    bgra.play_with_strategy(game_object=game, strategy=strategy_o, str_player=2)

    # print("\nDouczanie strategii o na ustalonej strategii x, by sprawdzić")
    # print("na ile uczenie równoczesne było skuteczne.\n")
    # _, strategy_o_doucz = \
    #     board_game_train_Q2(game,players_to_train = [2], strategy_x = strategy_x, number_of_games = 10000)
    # strategy_o_doucz.to_file("strategy_o_doucz.txt",)
    # print("test stategii douczanej o i uczonej x:")
    # num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, strategy_x,strategy_o_doucz)
    # game.print_test_to_file("gry_uczonej_X_z_douczana_O.txt",num_win_x, num_win_o, num_draws, Games, Rewards)


experiment_par_train()