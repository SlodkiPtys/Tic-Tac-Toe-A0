# functions supporting tictac project 
import numpy as np


# -----------------------------------------------------------
# Board games classes (tic-tac-toe, gomoku, renju, connect4, checkers, chess, ...)

# tic-tac-toe  first movement by player 1 - cross, second by player 2 - circle
# each board with 3x3 cells represents state , each cell can be empty (0), contain x (1) or contain o (2)
# e.g. for board A = [[0,2,0],[0,1,0], [0,0,0]]:
#          _ o _
#     A =  _ x _
#          _ _ _
class Tictactoe:
    def __init__(self):
        pass

    def initial_state(self):
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int)

    # reward value from 'x' player point of view only in terminal states (-1: lose, 0: draw, 1: win)
    def __reward(self, Board):
        R = 0
        if (np.max(np.sum(Board, 0)) == 6) | (np.max(np.sum(Board, 1)) == 6) | \
                (Board[0, 0] + Board[1, 1] + Board[2, 2] == 6) | (Board[0, 2] + Board[1, 1] + Board[2, 0] == 6):
            R = -1
        if (R == 0):
            B = Board % 2
            if (np.max(np.sum(B, 0)) == 3) | (np.max(np.sum(B, 1)) == 3) | \
                    (B[0, 0] + B[1, 1] + B[2, 2] == 3) | (B[0, 2] + B[1, 1] + B[2, 0] == 3):
                R = 1
        return R

    def end_of_game(self, _R=0, _number_of_moves=0, _Board=[], _action_nr=0):
        if (np.abs(_R) > 0) | (_number_of_moves >= 9):
            return True
        else:
            return False

    # output: player, list of states after possible moves, rewards for moves
    def actions(self, State, player=0):
        actions = []

        Board = State
        empty_cells = np.where(Board == 0)
        empty_cells_number = len(empty_cells[0])

        for i in range(empty_cells_number):
            row = empty_cells[0][i]
            column = empty_cells[1][i]
            actions.append([row, column])

        return actions

    def next_state_and_reward(self, player, State, action):
        row, col = action
        NextState = np.copy(State)
        NextState[row, col] = player
        reward = self.__reward(NextState)
        return NextState, reward

    def state_key(self, State):
        return str(State)

    # printing to text file info about test results and particular games (each game in a row)
    def print_test_to_file(self, filename, num_win_x, num_win_o, num_draws, Games, Rewards):
        f = open(filename, "w")
        number_of_games = len(Games)

        for g in range(number_of_games):
            Boards, Actions = Games[g]
            num_rows, num_col = np.shape(Boards[0])
            num_of_boards = len(Boards)
            result = " draw"
            if Rewards[g] == 1:
                result = " x win"
            elif Rewards[g] == -1:
                result = " o win"
            f.write("game " + str(g) + result + ":\n")
            for r in range(num_rows):
                row = ""
                for b in range(num_of_boards):
                    A = Boards[b]
                    for c in range(num_col):
                        if A[r, c] == 0:
                            row += "_"
                        elif A[r, c] == 1:
                            row += "x"
                        elif A[r, c] == 2:
                            row += "o"
                    row += "  "
                f.write(row + "\n")
            f.write("\n")

        print("results after %d games: " % (number_of_games))
        print("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.write("results after %d games: " % (number_of_games))
        f.write("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.close()

    def move_verification(self, State, actions, NextState, player, f):
        pass

    # end of class Tictactoe


# general cross and circles game: board with given size + number of stones in a row
# as w win/loss condition
# (vertical, horizontal, diagonally) + information if new stones need to be adjecent
# to existing ones (if adjacent - also first move in the center of a board to simplify training of
# an approximator)
class Tictac_general:

    def __init__(self, num_of_rows, num_of_columns, num_of_stones, if_adjacent):
        self.num_of_rows = num_of_rows
        self.num_of_columns = num_of_columns
        self.num_of_stones = num_of_stones
        self.if_adjacent = if_adjacent

    def initial_state(self):
        return np.zeros([self.num_of_rows, self.num_of_columns], dtype=int)

    # quicker version of reward which takes into account only
    # pieses in a vertical, horizontel and diagonal / \ sequences
    # contained piece after current move
    def __reward_after_move(self, A, row, column, player):
        R = 0
        # horizontal direction:
        i = 0
        while True:
            if column + i + 1 < self.num_of_columns:
                if A[row, column + i + 1] == player:
                    i += 1
                else:
                    break
            else:
                break
        num_of_pieces = i + 1
        i = 0
        while True:
            if column - i - 1 >= 0:
                if A[row, column - i - 1] == player:
                    i += 1
                else:
                    break
            else:
                break
        num_of_pieces += i
        if num_of_pieces >= self.num_of_stones:
            R = (player == 1) - (player == 2)

        if R == 0:
            # vertical direction:
            i = 0
            while True:
                if row + i + 1 < self.num_of_rows:
                    if A[row + i + 1, column] == player:
                        i += 1
                    else:
                        break
                else:
                    break
            num_of_pieces = i + 1
            i = 0
            while True:
                if row - i - 1 >= 0:
                    if A[row - i - 1, column] == player:
                        i += 1
                    else:
                        break
                else:
                    break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)

        if R == 0:
            # diagonal \ direction:
            i = 0
            while True:
                if (row + i + 1 < self.num_of_rows) & (column + i + 1 < self.num_of_columns):
                    if A[row + i + 1, column + i + 1] == player:
                        i += 1
                    else:
                        break
                else:
                    break
            num_of_pieces = i + 1
            i = 0
            while True:
                if (row - i - 1 >= 0) & (column - i - 1 >= 0):
                    if A[row - i - 1, column - i - 1] == player:
                        i += 1
                    else:
                        break
                else:
                    break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)

        if R == 0:
            # diagonal / direction:
            i = 0
            while True:
                if (row + i + 1 < self.num_of_rows) & (column - i - 1 >= 0):
                    if A[row + i + 1, column - i - 1] == player:
                        i += 1
                    else:
                        break
                else:
                    break
            num_of_pieces = i + 1
            i = 0
            while True:
                if (row - i - 1 >= 0) & (column + i + 1 < self.num_of_columns):
                    if A[row - i - 1, column + i + 1] == player:
                        i += 1
                    else:
                        break
                else:
                    break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)
        return R

    # checking if end of game (draw)
    def end_of_game(self, _R=0, _number_of_moves=0, _Board=[], _action_nr=0):
        if (np.abs(_R) > 0) | (_number_of_moves >= self.num_of_rows * self.num_of_columns):
            return True
        else:
            return False

    # output: player, list of states after possible moves, rewards for moves
    def actions(self, A, player=0):
        actions = []

        empty_cells = np.where(A == 0)
        empty_cells_number = len(empty_cells[0])
        number_of_pieces = self.num_of_rows * self.num_of_columns - empty_cells_number

        if self.if_adjacent:
            if (self.if_adjacent) & (number_of_pieces == 0):
                actions.append([self.num_of_rows // 2, self.num_of_columns // 2])
            elif (self.if_adjacent) & (number_of_pieces == 1):
                actions.append([self.num_of_rows // 2 - 1, self.num_of_columns // 2])
                actions.append([self.num_of_rows // 2 - 1, self.num_of_columns // 2 - 1])
            else:
                for i in range(empty_cells_number):
                    row = empty_cells[0][i]
                    column = empty_cells[1][i]
                    if self.if_adjacent:
                        num_of_neibours = 0
                        for r in range(3):
                            for c in range(3):
                                rr = row + r - 1
                                cc = column + c - 1
                                if (rr >= 0) & (rr < self.num_of_rows) & (cc >= 0) & (cc < self.num_of_columns):
                                    num_of_neibours += (A[rr, cc] != 0)
                    if empty_cells_number == self.num_of_rows * self.num_of_columns:
                        num_of_neibours = 1
                    if (self.if_adjacent == False) | (num_of_neibours > 0):
                        actions.append([row, column])
        else:
            for i in range(empty_cells_number):
                row = empty_cells[0][i]
                column = empty_cells[1][i]
                actions.append([row, column])

        return actions

    def next_state_and_reward(self, player, State, action):
        row, col = action
        NextState = np.copy(State)
        NextState[row, col] = player
        reward = self.__reward_after_move(NextState, row, col, player)
        return NextState, reward

    def state_key(self, State):
        return str(State)

    # printing to text file info about test results and particular games (each game in a row)
    def print_test_to_file(self, filename, num_win_x, num_win_o, num_draws, Games, Rewards):
        f = open(filename, "w")
        number_of_games = len(Games)

        for g in range(number_of_games):
            Boards, Actions = Games[g]
            num_rows, num_col = np.shape(Boards[0])
            num_of_boards = len(Boards)
            result = " draw"
            if Rewards[g] == 1:
                result = " x win"
            elif Rewards[g] == -1:
                result = " o win"
            f.write("game " + str(g) + result + ":\n")
            for r in range(num_rows):
                row = ""
                for b in range(num_of_boards):
                    A = Boards[b]
                    for c in range(num_col):
                        if A[r, c] == 0:
                            row += "_"
                        elif A[r, c] == 1:
                            row += "x"
                        elif A[r, c] == 2:
                            row += "o"
                    row += "  "
                f.write(row + "\n")
            f.write("\n")

        print("results after %d games: " % (number_of_games))
        print("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.write("results after %d games: " % (number_of_games))
        f.write("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.close()

    def move_verification(self, State, actions, NextState, player, f):
        pass

    # end of class Tictac_general


class Connect4(Tictac_general):
    def __init__(self):
        super().__init__(6, 7, 4, False)

    def actions(self, A, player=0):
        actions = []
        for c in range(self.num_of_columns):
            r = self.num_of_rows - 1
            while (r >= 0):
                if A[r, c] == 0:
                    actions.append([r, c])
                    break
                r -= 1
        return actions


# Two end modes: with kings -> the player wins when king is mated, without kings ->
# the player wins after removing all opponens pieces
class Chess():
    Pawn = 1  # static variables (belonging to class)
    Knight = 2  # white pieces identifies
    Bishop = 3
    Rook = 4
    Queen = 5
    King = 6

    BlackShift = 1000  # shift of black pieces identifies

    Target_BlockEnemy = 0  # geme target: block enemy to win
    Target_CheckmateKing = 1  # game target: checkmate opponets king to win (bloking enemy provides pat!)

    class ChessState():
        def __init__(self, _Board=None, _white_king_checked=None, _black_king_checked=None, \
                     _white_king_mated=None, _black_king_mated=None, \
                     _white_blocked=None, _black_blocked=None, \
                     _white_small_castling_possible=None, _black_small_castling_possible=None, \
                     _white_big_castling_possible=None, _black_big_castling_possible=None, \
                     _white_2pawn_column=None, _black_2pawn_column=None):
            self.Board = _Board
            self.white_king_checked = _white_king_checked
            self.black_king_checked = _black_king_checked
            self.white_king_mated = _white_king_mated
            self.black_king_mated = _black_king_mated
            self.white_blocked = _white_blocked
            self.black_blocked = _black_blocked
            self.white_small_castling_possible = _white_small_castling_possible
            self.black_small_castling_possible = _black_small_castling_possible
            self.white_big_castling_possible = _white_big_castling_possible
            self.black_big_castling_possible = _black_big_castling_possible
            self.white_2pawn_column = _white_2pawn_column
            self.black_2pawn_column = _black_2pawn_column

    class Move():
        def __init__(self, _piece_type, _r_from, _r_to, _c_from, _c_to, _piece_to_promote=None) -> None:
            self.piece_type = _piece_type
            self.r_from = _r_from
            self.r_to = _r_to
            self.c_from = _c_from
            self.c_to = _c_to
            self.piece_to_promote = _piece_to_promote

    # initial board from text file
    # The standard initial board can looks like this:
    #    a b c d e f g h
    #    ----------------
    # 8 |W S G H K G S W | 8
    # 7 |P P P P P P P P | 7
    # 6 |. . . . . . . . | 6
    # 5 |. . . . . . . . | 5
    # 4 |. . . . . . . . | 4
    # 3 |. . . . . . . . | 3
    # 2 |p p p p p p p p | 2
    # 1 |w s g h k g s w | 1
    #    ----------------
    #    a b c d e f g h
    #
    # description: W - rook, S - knight, G - bishop, H - queen, K - king, P - pawn, '.' - empty square
    # white pieces by small letters, black pieces by capital letters
    # row and column numbers can be omitted, rows are recognized by pipelines -  '|'
    # pieces are separated by spaces
    def __read_board(self, filename):
        num_of_rows = 8
        num_of_columns = 8
        __Board = np.zeros([num_of_rows, num_of_columns], dtype=int)
        pieces_dict = {'.': 0, 'p': self.Pawn, 'P': self.Pawn + self.BlackShift, \
                       's': self.Knight, 'S': self.Knight + self.BlackShift, \
                       'g': self.Bishop, 'G': self.Bishop + self.BlackShift, \
                       'w': self.Rook, 'W': self.Rook + self.BlackShift, \
                       'h': self.Queen, 'H': self.Queen + self.BlackShift, \
                       'k': self.King, 'K': self.King + self.BlackShift}

        f = open(filename, 'r').read()
        lines = f.split('\n')
        number_of_lines = lines.__len__() - 1
        nums_of_pieces = []
        for i in range(number_of_lines):
            lin_split = lines[i].split('|')
            if len(lin_split) == 3:
                row_split = lin_split[1].split(' ')
                __num_of_col = 0
                for j in range(len(row_split)):
                    if row_split[j] != '':
                        __num_of_col += 1
                nums_of_pieces.append(__num_of_col)

        # sprawdzenie czy liczby kolumn są takie same w każdym wierszu:
        # ...............

        __Board = np.zeros([len(nums_of_pieces), __num_of_col], dtype=int)
        __num_of_rows = 0
        for i in range(number_of_lines):
            lin_split = lines[i].split('|')
            if len(lin_split) == 3:
                row_split = lin_split[1].split(' ')
                __num_of_col = 0
                for j in range(len(row_split)):
                    if row_split[j] != '':
                        __Board[__num_of_rows, __num_of_col] = pieces_dict[row_split[j]]
                        __num_of_col += 1
                __num_of_rows += 1
                # print("row_split = " +  str(row_split) + " len = " + str(len(row_split)))
        # f.close()
        return __Board

    def __init__(self, filename=""):
        if len(filename) > 0:
            self.InitialBoard = self.__read_board(filename)
            # self.InitialBoard = np.copy(Board)
        else:
            self.InitialBoard = np.zeros([8, 8], dtype=int)
            for c in range(8):
                self.InitialBoard[6, c] = self.Pawn
                self.InitialBoard[1, c] = self.Pawn + self.BlackShift
            self.InitialBoard[7, 0] = self.InitialBoard[7, 7] = self.Rook
            self.InitialBoard[0, 0] = self.InitialBoard[0, 7] = self.Rook + self.BlackShift
            self.InitialBoard[7, 1] = self.InitialBoard[7, 6] = self.Knight
            self.InitialBoard[0, 1] = self.InitialBoard[0, 6] = self.Knight + self.BlackShift
            self.InitialBoard[7, 2] = self.InitialBoard[7, 5] = self.Bishop
            self.InitialBoard[0, 2] = self.InitialBoard[0, 5] = self.Bishop + self.BlackShift
            self.InitialBoard[7, 3] = self.Queen
            self.InitialBoard[0, 3] = self.Queen + self.BlackShift
            self.InitialBoard[7, 4] = self.King
            self.InitialBoard[0, 4] = self.King + self.BlackShift
        # Two types of end-game rules:
        # if king appears at the start - the game is finished after mating the king
        # if king disappear - the game is over any move is impossible for a player
        # e.g. after capturing all pieces

        self.num_of_rows, self.num_of_columns = np.shape(self.InitialBoard)
        num_of_black_kings = 0
        num_of_white_kings = 0
        for r in range(self.num_of_rows):
            for c in range(self.num_of_columns):
                if self.InitialBoard[r, c] == self.King:
                    num_of_white_kings += 1
                elif self.InitialBoard[r, c] == self.King + self.BlackShift:
                    num_of_black_kings += 1
        if num_of_white_kings > 0:
            self.black_objective = self.Target_CheckmateKing
        else:
            self.black_objective = self.Target_BlockEnemy
        if num_of_black_kings > 0:
            self.white_objective = self.Target_CheckmateKing
        else:
            self.white_objective = self.Target_BlockEnemy

    def initial_state(self):
        _white_king_checked = False
        _black_king_checked = False
        _white_king_mated = False
        _black_king_mated = False
        _white_blocked = False
        _black_blocked = False
        if (self.num_of_rows == 8) & (self.num_of_columns == 8):
            _white_small_castling_possible = True
            _black_small_castling_possible = True
            _white_big_castling_possible = True
            _black_big_castling_possible = True
        else:
            _white_small_castling_possible = False
            _black_small_castling_possible = False
            _white_big_castling_possible = False
            _black_big_castling_possible = False
        _white_2pawn_column = None
        _black_2pawn_column = None
        IniState = self.ChessState(self.InitialBoard, _white_king_checked, _black_king_checked, \
                                   _white_king_mated, _black_king_mated, \
                                   _white_blocked, _black_blocked, \
                                   _white_small_castling_possible, _black_small_castling_possible, \
                                   _white_big_castling_possible, _black_big_castling_possible, \
                                   _white_2pawn_column, _black_2pawn_column)

        return IniState

    # quicker version of reward which takes into account only
    # pieses in a vertical, horizontel and diagonal / \ sequences
    # contained piece after current move
    # def reward_after_move(self, A, row, column, player):
    #    R = 0
    #    return R

    # checking if end of game (draw)
    def end_of_game(self, _R=0, _number_of_moves=0, State=[], _action_nr=0):
        if (np.abs(_R) > 0.95) | (_number_of_moves >= 10 * self.num_of_rows * self.num_of_columns) | \
                (State.white_king_mated == True) | (State.black_king_mated == True) | \
                (State.white_blocked == True) | (State.black_blocked == True):
            return True
        else:
            return False

    def state_key(self, State):
        return str(State.Board)

        # sprawdzenie czy pole jest szachowane przez figurę gracza <by_player>:

    # by_player == 1 - białe szachują pole
    # by_player == 2 - czarne szachują pole
    def __list_of_checking_figures(self, Board, position, by_player):
        r, c = position
        shift = 0  # shift of piece identifier
        up = -1  # pawn moving direction
        if by_player == 2:
            shift = self.BlackShift
            up = 1
        shift_opo = self.BlackShift - shift
        list_of_checking_figures = []
        # check from by_player pawns:
        if (by_player == 1) & (r < self.num_of_rows - 1):
            if c > 0:
                if Board[r + 1, c - 1] == self.Pawn + shift:
                    list_of_checking_figures.append([r + 1, c - 1])
            if c < self.num_of_columns - 1:
                if Board[r + 1, c + 1] == self.Pawn + shift:
                    list_of_checking_figures.append([r + 1, c + 1])
        elif (by_player == 2) & (r > 0):
            if c > 0:
                if Board[r - 1, c - 1] == self.Pawn + shift:
                    list_of_checking_figures.append([r - 1, c - 1])
            if c < self.num_of_columns - 1:
                if Board[r - 1, c + 1] == self.Pawn + shift:
                    list_of_checking_figures.append([r - 1, c + 1])

        # check from by_player knights:
        for dr, dc in [[-1, 2], [1, 2], [-1, -2], [1, -2], [2, -1], [2, 1], [-2, -1], [-2, 1]]:
            r2 = r + dr
            c2 = c + dc
            if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                if (Board[r2, c2] == self.Knight + shift):
                    list_of_checking_figures.append([r2, c2])

        # check from by_player's bishops, queens or king - in diagonal directions:
        for dr, dc in [[-1, -1], [-1, 1], [1, -1], [1, 1]]:
            if_end = False  # end of file loop flag
            r2 = r
            c2 = c
            while not if_end:
                r2 += dr
                c2 += dc
                if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                    if ((Board[r2, c2] == self.Bishop + shift) \
                            | (Board[r2, c2] == self.Queen + shift) \
                            | ((Board[r2, c2] == self.King + shift) & ((r2 - r) * (r2 - r) == 1))):
                        list_of_checking_figures.append([r2, c2])
                        if_end = True
                    elif Board[r2, c2] != 0:
                        if_end = True
                else:  # out of board
                    if_end = True

        # check from opponent's rooks or queens - in horizontal or vertical direction:
        for dr, dc in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            if_end = False  # end of file loop flag
            r2 = r
            c2 = c
            while not if_end:
                r2 += dr
                c2 += dc
                if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                    if ((Board[r2, c2] == self.Rook + shift) \
                            | (Board[r2, c2] == self.Queen + shift) \
                            | ((Board[r2, c2] == self.King + shift) \
                               & (((r2 - r) * (r2 - r) == 1) | ((c2 - c) * (c2 - c) == 1)))):
                        list_of_checking_figures.append([r2, c2])
                        if_end = True
                    elif Board[r2, c2] != 0:
                        if_end = True
                else:  # out of board
                    if_end = True

        # check from other non-standard figures:
        # .....................................
        # .....................................
        return list_of_checking_figures

    def list_of_checking_figures(self, Board, position, by_player):
        return self.__list_of_checking_figures(Board, position, by_player)

    # sprawdzenie czy pole jest szachowane przez figurę gracza <by_player>
    # oraz czy byłoby szachowane, gdyby usunąć figurę przeciwnego koloru
    # (można w ten sposób znaleźć ruchy odsłanianiające króla)
    # by_player == 1 - białe szachują pole
    # by_player == 2 - czarne szachują pole
    def __lists_of_extended_checking_figures(self, Board, position, by_player):
        r, c = position
        shift = 0  # shift of piece identifier
        up = -1  # pawn moving direction
        if by_player == 2:
            shift = self.BlackShift
            up = 1
        shift_opo = self.BlackShift - shift
        list_of_checking_figures = []
        list_of_checking_by_obstacle = []
        # check from by_player pawns:
        if (by_player == 1) & (r < self.num_of_rows - 1):
            if c > 0:
                if Board[r + 1, c - 1] == self.Pawn + shift:
                    list_of_checking_figures.append([r + 1, c - 1])
            if c < self.num_of_columns - 1:
                if Board[r + 1, c + 1] == self.Pawn + shift:
                    list_of_checking_figures.append([r + 1, c + 1])
        elif (by_player == 2) & (r > 0):
            if c > 0:
                if Board[r - 1, c - 1] == self.Pawn + shift:
                    list_of_checking_figures.append([r - 1, c - 1])
            if c < self.num_of_columns - 1:
                if Board[r - 1, c + 1] == self.Pawn + shift:
                    list_of_checking_figures.append([r - 1, c + 1])

        # check from by_player knights:
        for dr, dc in [[-1, 2], [1, 2], [-1, -2], [1, -2], [2, -1], [2, 1], [-2, -1], [-2, 1]]:
            r2 = r + dr
            c2 = c + dc
            if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                if (Board[r2, c2] == self.Knight + shift):
                    list_of_checking_figures.append([r2, c2])

        # check from by_player's bishops, queens or king - in diagonal directions:
        for dr, dc in [[-1, -1], [-1, 1], [1, -1], [1, 1]]:
            if_end = False  # end of file loop flag
            own_obstacle_pos = []  # potential position of own piece between checking position
            # and opponent piece
            r2 = r
            c2 = c
            while not if_end:
                r2 += dr
                c2 += dc
                if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                    if ((Board[r2, c2] == self.Bishop + shift) \
                            | (Board[r2, c2] == self.Queen + shift) \
                            | ((Board[r2, c2] == self.King + shift) & ((r2 - r) * (r2 - r) == 1))):
                        if own_obstacle_pos == []:
                            list_of_checking_figures.append([r2, c2])
                        else:
                            list_of_checking_by_obstacle.append([r2, c2, own_obstacle_pos[0], own_obstacle_pos[1]])
                        if_end = True
                    elif (own_obstacle_pos == []) & (Board[r2, c2] != 0) & \
                            (Board[r2, c2] > shift_opo) & (Board[r2, c2] < shift_opo + self.BlackShift):
                        own_obstacle_pos = [r2, c2]  # own figure in current position
                    elif Board[r2, c2] != 0:
                        if_end = True
                else:  # out of board
                    if_end = True

        # check from opponent's rooks or queens - in horizontal or vertical direction:
        for dr, dc in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            if_end = False  # end of file loop flag
            own_obstacle_pos = []  # potential position of own piece between checking position
            # and opponent piece
            r2 = r  # current position coordinates
            c2 = c
            while not if_end:
                r2 += dr
                c2 += dc
                if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                    if ((Board[r2, c2] == self.Rook + shift) \
                            | (Board[r2, c2] == self.Queen + shift) \
                            | ((Board[r2, c2] == self.King + shift) \
                               & (((r2 - r) * (r2 - r) == 1) | ((c2 - c) * (c2 - c) == 1)))):
                        if own_obstacle_pos == []:
                            list_of_checking_figures.append([r2, c2])
                        else:
                            list_of_checking_by_obstacle.append([r2, c2, own_obstacle_pos[0], own_obstacle_pos[1]])
                        if_end = True
                    elif (own_obstacle_pos == []) & (Board[r2, c2] != 0) & \
                            (Board[r2, c2] > shift_opo) & (Board[r2, c2] < shift_opo + self.BlackShift):
                        own_obstacle_pos = [r2, c2]  # own figure in current position
                    elif Board[r2, c2] != 0:
                        if_end = True
                else:  # out of board
                    if_end = True

        # check from other non-standard figures:
        # .....................................
        # .....................................
        return list_of_checking_figures, list_of_checking_by_obstacle

    # realizacja ruchu przy założeniu, że ruch jest dopuszczalny:
    def next_state_and_reward(self, player, State, action):
        player_opo = 3 - player
        if player == 1:
            player_objective = self.white_objective
            shift = 0
        else:
            player_objective = self.black_objective
            shift = self.BlackShift

        shift_opo = self.BlackShift - shift

        # making action:
        NextState = self.ChessState()
        NextState.Board = np.copy(State.Board)
        NextState.white_big_castling_possible = State.white_big_castling_possible
        NextState.white_small_castling_possible = State.white_small_castling_possible
        NextState.black_big_castling_possible = State.black_big_castling_possible
        NextState.black_small_castling_possible = State.black_small_castling_possible
        NextState.white_king_checked = False
        NextState.black_king_checked = False

        promotion = None
        if len(action) == 5:
            piece, r_from, c_from, r_to, c_to = action
        elif len(action) == 6:
            piece, r_from, c_from, r_to, c_to, promotion = action
        NextState.Board[r_from, c_from] = 0
        if promotion != None:
            NextState.Board[r_to, c_to] = promotion
        else:
            NextState.Board[r_to, c_to] = piece
        # short castling (roszada mała):
        # to czy roszada była dozwolona powinno być sprawdzone w actions()
        if (piece == self.King) & (r_from == 7) & (c_from == 4) & (r_to == 7) & (c_to == 6):
            NextState.Board[7, 7] = 0
            NextState.Board[7, 5] = self.Rook
        elif (piece == self.King + self.BlackShift) & (r_from == 0) & (c_from == 4) & (r_to == 0) & (c_to == 6):
            NextState.Board[0, 7] = 0
            NextState.Board[0, 5] = self.Rook + self.BlackShift
        # long castling (roszada długa):
        elif (piece == self.King) & (r_from == 7) & (c_from == 4) & (r_to == 7) & (c_to == 2):
            NextState.Board[7, 0] = 0
            NextState.Board[7, 3] = self.Rook
        elif (piece == self.King + self.BlackShift) & (r_from == 0) & (c_from == 4) & (r_to == 0) & (c_to == 2):
            NextState.Board[0, 7] = 0
            NextState.Board[0, 3] = self.Rook + self.BlackShift

            # verification if castling is still possible:
        if (piece == self.Rook) & (r_from == 7) & (c_from == 0):
            NextState.white_big_castling_possible = False
        elif (piece == self.Rook) & (r_from == 7) & (c_from == 7):
            NextState.white_small_castling_possible = False
        elif (piece == self.Rook + self.BlackShift) & (r_from == 0) & (c_from == 0):
            NextState.black_big_castling_possible = False
        elif (piece == self.Rook + self.BlackShift) & (r_from == 0) & (c_from == 7):
            NextState.black_small_castling_possible = False
        elif piece == self.King:
            NextState.white_big_castling_possible = False
            NextState.white_small_castling_possible = False
        elif piece == self.King + self.BlackShift:
            NextState.black_big_castling_possible = False
            NextState.black_small_castling_possible = False

        Reward = 0
        if player_objective == self.Target_CheckmateKing:
            # finding opposite king position:
            for r in range(self.num_of_rows):
                for c in range(self.num_of_columns):
                    if NextState.Board[r, c] == self.King + shift_opo:
                        opo_king_pos = [r, c]
                        break
            # finding if opposite king is checked:
            list_of_checking = self.__list_of_checking_figures(NextState.Board, opo_king_pos, player)
            list_of_opo_actions = self.actions(NextState, player_opo)

            if len(list_of_checking) > 0:
                if player == 1:
                    NextState.black_king_checked = True
                else:
                    NextState.white_king_checked = True

                # finding if opposite king is checkmated:
                if len(list_of_opo_actions) == 0:
                    if player == 1:
                        Reward = 1
                        NextState.black_king_mated = True
                        # print("checkmate against black!")
                    else:
                        Reward = -1
                        NextState.white_king_mated = True
                        # print("checkmate against white!")
            elif len(list_of_opo_actions) == 0:
                if player == 1:
                    NextState.black_blocked = True
                else:
                    NextState.white_blocked = True


        elif player_objective == self.Target_BlockEnemy:
            list_of_opo_actions = self.actions(NextState, player_opo)
            if len(list_of_opo_actions) == 0:
                if player == 1:
                    Reward = 1
                    NextState.black_blocked = True
                else:
                    Reward = -1
                    NextState.white_blocked = True

        # output: player, list of states after possible moves, rewards for moves:
        return NextState, Reward

    def actions(self, State, player):
        Board = State.Board
        shift = 0  # shift of piece identifier
        up = -1  # pawn moving direction
        if player == 2:
            shift = self.BlackShift
            up = 1
        player_opo = 3 - player
        shift_opo = self.BlackShift - shift
        num_of_own_kings = 0  # number of player's kings
        own_king_pos = []  # position of king (relevant if one king exactly)

        moves_potential = []  # potential moves without checking taken into account
        for r in range(self.num_of_rows):
            for c in range(self.num_of_columns):
                if Board[r, c] == 0:
                    pass  # the most of cells are empty!
                elif Board[r, c] == self.Pawn + shift:  # Pawn moves
                    rup = r + up
                    if (rup >= 0) & (rup < self.num_of_rows):  # one field up possible
                        if Board[rup, c] == 0:
                            if (rup == 0) | (rup == self.num_of_rows - 1):  # moves with promotion
                                moves_potential.append([self.Pawn + shift, r, c, rup, c, self.Knight + shift])
                                moves_potential.append([self.Pawn + shift, r, c, rup, c, self.Bishop + shift])
                                moves_potential.append([self.Pawn + shift, r, c, rup, c, self.Rook + shift])
                                moves_potential.append([self.Pawn + shift, r, c, rup, c, self.Queen + shift])
                            else:  # only move
                                moves_potential.append([self.Pawn + shift, r, c, rup, c])
                        for i in [-1, 1]:
                            c2 = c + i
                            if (c2 >= 0) & (c2 < self.num_of_columns):
                                if (Board[rup, c2] > shift_opo) & (Board[rup, c2] < shift_opo + self.BlackShift):
                                    if (rup == 0) | (rup == self.num_of_rows - 1):  # beating with promotion
                                        moves_potential.append([self.Pawn + shift, r, c, rup, c2, self.Knight + shift])
                                        moves_potential.append([self.Pawn + shift, r, c, rup, c2, self.Bishop + shift])
                                        moves_potential.append([self.Pawn + shift, r, c, rup, c2, self.Rook + shift])
                                        moves_potential.append([self.Pawn + shift, r, c, rup, c2, self.Queen + shift])
                                    else:  # only beating opponent piece
                                        moves_potential.append([self.Pawn + shift, r, c, rup, c2])
                    if ((r == 1) & (player == 2)) | ((r == self.num_of_rows - 2) & (player == 1)):
                        r2up = r + 2 * up
                        if (r2up >= 0) & (r2up < self.num_of_rows):
                            if (Board[rup, c] == 0) & (Board[r2up, c] == 0):
                                if (r2up == 0) | (r2up == self.num_of_rows - 1):  # moves with promotion
                                    moves_potential.append([self.Pawn + shift, r, c, r2up, c, self.Knight + shift])
                                    moves_potential.append([self.Pawn + shift, r, c, r2up, c, self.Bishop + shift])
                                    moves_potential.append([self.Pawn + shift, r, c, r2up, c, self.Rook + shift])
                                    moves_potential.append([self.Pawn + shift, r, c, r2up, c, self.Queen + shift])
                                else:  # only move
                                    moves_potential.append([self.Pawn + shift, r, c, r2up, c])

                    # .................... bicie w przelocie (en passant)  może odsłonić króla
                    # end of Pawn moves
                elif Board[r, c] == self.Knight + shift:
                    for dr, dc in [[-1, 2], [1, 2], [-1, -2], [1, -2], [2, -1], [2, 1], [-2, -1], [-2, 1]]:
                        r2 = r + dr
                        c2 = c + dc
                        if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                            if (Board[r2, c2] == 0) | (
                                    (Board[r2, c2] > shift_opo) & (Board[r2, c2] < shift_opo + self.BlackShift)):
                                moves_potential.append([self.Knight + shift, r, c, r2, c2])
                    # end of Knight moves
                elif (Board[r, c] == self.Bishop + shift) | (Board[r, c] == self.Rook + shift) | \
                        (Board[r, c] == self.Queen + shift):
                    directions = \
                        [[-1, -1, self.Bishop], [-1, 1, self.Bishop], [1, -1, self.Bishop], [1, 1, self.Bishop],
                         [-1, 0, self.Rook], [1, 0, self.Rook], [0, -1, self.Rook], [0, 1, self.Rook],
                         ]
                    for dr, dc, piece in directions:
                        if (Board[r, c] == piece + shift) | (Board[r, c] == self.Queen + shift):
                            if_end = False
                            r2 = r
                            c2 = c
                            while not if_end:
                                r2 += dr
                                c2 += dc
                                if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                                    if Board[r2, c2] == 0:  # empty cell
                                        moves_potential.append([Board[r, c], r, c, r2, c2])
                                    elif (Board[r2, c2] > shift_opo) & (Board[r2, c2] < shift_opo + self.BlackShift):
                                        moves_potential.append([Board[r, c], r, c, r2, c2])
                                        if_end = True
                                    else:
                                        if_end = True
                                else:
                                    if_end = True
                    # end of Bishop, Rook or Queen moves
                elif Board[r, c] == self.King + shift:
                    num_of_own_kings += 1
                    own_king_pos = [r, c]
                    king_checking_pieces, king_checking_pieces_by_obstacle = \
                        self.__lists_of_extended_checking_figures(Board, own_king_pos, player_opo)

                    for dr, dc in [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]:
                        r2 = r + dr
                        c2 = c + dc
                        Board[r, c] = 0  # by król nie zasłaniał szachowanego pola
                        if (r2 >= 0) & (r2 < self.num_of_rows) & (c2 >= 0) & (c2 < self.num_of_columns):
                            if (Board[r2, c2] == 0) | (
                                    (Board[r2, c2] > shift_opo) & (Board[r2, c2] < shift_opo + self.BlackShift)):
                                pos_checking_from = self.__list_of_checking_figures(Board, [r2, c2], player_opo)
                                if len(pos_checking_from) == 0:
                                    moves_potential.append([self.King + shift, r, c, r2, c2])
                        Board[r, c] = self.King + shift

                    # castling:
                    if (len(king_checking_pieces) == 0) & (State.white_small_castling_possible == True):
                        if (player == 1) & ([r, c] == [7, 4]) & \
                                (Board[7, 7] == self.Rook) & (Board[7, 5] == 0) & (Board[7, 6] == 0):
                            list_of_checking_figures = self.__list_of_checking_figures(Board, [7, 5], player_opo)
                            if len(list_of_checking_figures) == 0:
                                list_of_checking_figures = self.__list_of_checking_figures(Board, [7, 6], player_opo)
                                if len(list_of_checking_figures) == 0:
                                    moves_potential.append([self.King + shift, r, c, 7, 6])
                    if (len(king_checking_pieces) == 0) & (State.white_big_castling_possible == True):
                        if (player == 1) & ([r, c] == [7, 4]) & \
                                (Board[7, 0] == self.Rook) & (Board[7, 1] == 0) & (Board[7, 2] == 0) & (
                                Board[7, 3] == 0):
                            list_of_checking_figures = self.__list_of_checking_figures(Board, [7, 2], player_opo)
                            if len(list_of_checking_figures) == 0:
                                list_of_checking_figures = self.__list_of_checking_figures(Board, [7, 3], player_opo)
                                if len(list_of_checking_figures) == 0:
                                    moves_potential.append([self.King + shift, r, c, 7, 2])
                    if (len(king_checking_pieces) == 0) & (State.black_small_castling_possible == True):
                        if (player == 2) & ([r, c] == [0, 4]) & \
                                (Board[0, 7] == self.Rook + self.BlackShift) & (Board[0, 5] == 0) & (Board[0, 6] == 0):
                            list_of_checking_figures = self.__list_of_checking_figures(Board, [0, 5], player_opo)
                            if len(list_of_checking_figures) == 0:
                                list_of_checking_figures = self.__list_of_checking_figures(Board, [0, 6], player_opo)
                                if len(list_of_checking_figures) == 0:
                                    moves_potential.append([self.King + shift, r, c, 0, 6])
                    if (len(king_checking_pieces) == 0) & (State.black_big_castling_possible == True):
                        if (player == 2) & ([r, c] == [0, 4]) & \
                                (Board[0, 0] == self.Rook) & (Board[0, 1] == 0) & (Board[0, 2] == 0) & (
                                Board[0, 3] == 0):
                            list_of_checking_figures = self.__list_of_checking_figures(Board, [0, 2], player_opo)
                            if len(list_of_checking_figures) == 0:
                                list_of_checking_figures = self.__list_of_checking_figures(Board, [0, 3], player_opo)
                                if len(list_of_checking_figures) == 0:
                                    moves_potential.append([self.King + shift, r, c, 0, 2])
                                    # end of King moves

        # removing moves after which king is checked:
        moves_potential_new = []
        if num_of_own_kings == 1:
            rK, cK = own_king_pos  # own king position
            # tworzenie listy pozycji własnych figur, które mogą odsłonić króla
            # schodząc z linii szachowania:
            own_obstacle_poz = []
            for po in king_checking_pieces_by_obstacle:
                own_obstacle_poz.append([po[2], po[3]])
            if len(king_checking_pieces) == 0:
                # akceptacja wszystkich możliwych ruchów z wyjątkiem tych, które odsłaniają króla:
                for move in moves_potential:
                    if len(move) == 5:
                        piece, r_from, c_from, r_to, c_to = move
                    elif len(move) == 6:
                        piece, r_from, c_from, r_to, c_to, promotion = move

                    if [r_from, c_from] not in own_obstacle_poz:
                        moves_potential_new.append(move)
                    else:
                        # odnalezienie pozycji figury szachującej:
                        r_ch = c_ch = -1
                        for po in king_checking_pieces_by_obstacle:
                            r, c = po[2], po[3]
                            if [r, c] == [r_from, c_from]:
                                r_ch, c_ch = po[0], po[1]
                                break
                        # utworzenie listy pól pomiędzy własnym królem a figurą szachującą:
                        empty_poz = []
                        num_of_poz = max(np.abs(r_ch - rK), np.abs(c_ch - cK))
                        dr = (r_ch - rK) / (num_of_poz)
                        dc = (c_ch - cK) / (num_of_poz)
                        for i in range(num_of_poz):
                            empty_poz.append([rK + (i + 1) * dr, cK + (i + 1) * dc])
                        if [r_to, c_to] in empty_poz:  # jeśli pole docelowe ruchu na linii szacha lub stoi
                            # w nim figura szachująca (jest zbijana)
                            moves_potential_new.append(move)
            elif len(king_checking_pieces) == 1:
                # akceptacja ruchów króla, ruchów zbijających szachującą figurę i ruchów zasłaniających
                # króla przed szachem wieży, gońca lub hetmana:

                # utworzenie listy pustych pól pomiędzy własnym królem a figurą szachującą:
                r_ch, c_ch = king_checking_pieces[0]
                piece_ch = Board[r_ch, c_ch]
                empty_poz = []
                if (piece_ch == self.Queen + shift_opo) | (piece_ch == self.Bishop + shift_opo) | \
                        (piece_ch == self.Rook + shift_opo):
                    num_of_poz = max(np.abs(r_ch - rK), np.abs(c_ch - cK)) - 1
                    dr = (r_ch - rK) / (num_of_poz + 1)
                    dc = (c_ch - cK) / (num_of_poz + 1)
                    for i in range(num_of_poz):
                        empty_poz.append([rK + (i + 1) * dr, cK + (i + 1) * dc])

                for move in moves_potential:
                    if len(move) == 5:
                        piece, r_from, c_from, r_to, c_to = move
                    elif len(move) == 6:
                        piece, r_from, c_from, r_to, c_to, promotion = move
                    checking_piece_pos = king_checking_pieces[0]
                    if piece == self.King + shift:
                        moves_potential_new.append(move)
                    elif ([r_to, c_to] == checking_piece_pos) | ([r_to, c_to] in empty_poz):
                        if [r_from, c_from] not in own_obstacle_poz:
                            moves_potential_new.append(move)

            else:  # wiele szachujących figur -> nie da się zbić czy zasłonić
                # akceptacja tylko ruchów króla:
                # zakładamy, że wcześniej wyznaczono ruchy króla w pola nieszachowane
                for move in moves_potential:
                    if move[0] == self.King + shift:
                        moves_potential_new.append(move)

            moves_potential = moves_potential_new

        return moves_potential

    # printing board with pieces using special chess marks  - needs to use
    # special function to open text file: f = open("filename.txt","w",encoding='utf-8')
    # some problems with different width comparing chess marks and other characters
    def print_chessboard_unicode(self, Board, file=None):
        # pieces = u"\u2654\u2655\u2656\u2657\u2658\u2659\u265A\u265B\u265C\u265D\u265E\u265F\n"
        marks = {self.King: "\u2654", self.Queen: "\u2655", self.Rook: "\u2656", \
                 self.Bishop: "\u2657", self.Knight: "\u2658", self.Pawn: "\u2659", \
                 self.King + self.BlackShift: "\u265A", self.Queen + self.BlackShift: "\u265B", \
                 self.Rook + self.BlackShift: "\u265C", \
                 self.Bishop + self.BlackShift: "\u265D", self.Knight + self.BlackShift: "\u265E", \
                 self.Pawn + self.BlackShift: "\u265F", 0: chr(ord("\u2654") - 87)}
        num_of_rows, num_of_columns = np.shape(Board)
        for r in range(num_of_rows):
            row = ""
            for c in range(num_of_columns):
                row = row + marks[Board[r, c]] + " "
            if file:
                file.write(row + "\n")
            else:
                print(row)

    def print_chessboard_unicode2(self, Board):
        import sys
        if sys.version_info.major == 3:
            import tkinter as tk, tkinter.font as tkFont
        else:
            pass
            # import Tkinter as tk, tkFont

        root = tk.Tk()
        text = tk.Text(root, tabs=(200,))
        vsb = tk.Scrollbar(root, command=text.yview)
        text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        font = tkFont.Font(family="Courier New", size=24)

        marks = {self.King: "\u2654", self.Queen: "\u2655", self.Rook: "\u2656", \
                 self.Bishop: "\u2657", self.Knight: "\u2658", self.Pawn: "\u2659", \
                 self.King + self.BlackShift: "\u265A", self.Queen + self.BlackShift: "\u265B", \
                 self.Rook + self.BlackShift: "\u265C", \
                 self.Bishop + self.BlackShift: "\u265D", self.Knight + self.BlackShift: "\u265E", \
                 self.Pawn + self.BlackShift: "\u265F", 0: "."}
        num_of_rows, num_of_columns = np.shape(Board)
        for r in range(num_of_rows):
            row = ""
            for c in range(num_of_columns):
                row = row + marks[Board[r, c]]
            text.insert("end", row + "\n")

        root.mainloop()

    def print_chessboard(self, Board, file=None):
        # pieces = u"\u2654\u2655\u2656\u2657\u2658\u2659\u265A\u265B\u265C\u265D\u265E\u265F\n"
        marks = {self.King: "k", self.Queen: "h", self.Rook: "w", \
                 self.Bishop: "g", self.Knight: "s", self.Pawn: "p", \
                 self.King + self.BlackShift: "K", self.Queen + self.BlackShift: "H", \
                 self.Rook + self.BlackShift: "W", \
                 self.Bishop + self.BlackShift: "G", self.Knight + self.BlackShift: "S", \
                 self.Pawn + self.BlackShift: "P", 0: "."}
        col_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
                    12: 'm', \
                    13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
                    24: 'y', \
                    25: 'z'}
        num_of_rows, num_of_columns = np.shape(Board)
        init_string = "   "
        if num_of_rows > 9: init_string += " "
        if file:
            file.write(init_string)
            for c in range(num_of_columns):
                file.write(col_dict[c] + ' ')
            file.write("\n")
            file.write(init_string)
            for c in range(num_of_columns):
                file.write("--")
            file.write("\n")

        for r in range(num_of_rows):
            row = ""
            for c in range(num_of_columns):
                row = row + marks[Board[r, c]] + " "
            if file:
                row_desc = str(num_of_rows - r)
                if (r < 10) & (num_of_rows > 9):
                    row_desc = ' ' + row_desc
                file.write(row_desc + " |" + row + "| " + str(num_of_rows - r) + "\n")
            else:
                print(row)
        if file:
            file.write(init_string)
            for c in range(num_of_columns):
                file.write("--")
            file.write("\n")
            file.write(init_string)
            for c in range(num_of_columns):
                file.write(col_dict[c] + ' ')
            file.write("\n")

    def action_desc(self, State, action):
        marks = {self.King: "k", self.Queen: "h", self.Rook: "w", \
                 self.Bishop: "g", self.Knight: "s", self.Pawn: "p", \
                 self.King + self.BlackShift: "K", self.Queen + self.BlackShift: "H", \
                 self.Rook + self.BlackShift: "W", \
                 self.Bishop + self.BlackShift: "G", self.Knight + self.BlackShift: "S", \
                 self.Pawn + self.BlackShift: "P", 0: "."}
        num_of_rows, num_of_columns = np.shape(State.Board)
        if len(action) == 5:
            piece, r_from, c_from, r_to, c_to = action
        elif len(action) == 6:
            piece, r_from, c_from, r_to, c_to, promotion = action
        col_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
                    12: 'm', \
                    13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
                    24: 'y', \
                    25: 'z'}
        desc_string = marks[piece] + ' ' + col_dict[c_from] + str(num_of_rows - r_from) + ':' + \
                      col_dict[c_to] + str(num_of_rows - r_to)
        if len(action) == 6:
            desc_string += ' promo ' + marks[promotion]
        return desc_string

    # printing to text file info about test results and particular games (each game in a row)
    def print_test_to_file(self, filename, num_win_x, num_win_o, num_draws, Games, Rewards):
        f = open(filename, "w")
        number_of_games = len(Games)

        for g in range(number_of_games):
            States, Actions = Games[g]
            result = " draw"
            if Rewards[g] == 1:
                result = " white win"
            elif Rewards[g] == -1:
                result = " black win"
            f.write("game " + str(g) + result + ":\n")

            f.write("Początkowe ustawienie:\n")
            self.print_chessboard(States[0].Board, f)
            f.write("\n")
            for step_number in range(len(Actions)):
                if step_number % 2 == 0:
                    gracz = "białe"
                else:
                    gracz = "czarne"
                f.write(str(step_number) + ": " + gracz + " wybrały ruch " +
                        self.action_desc(States[step_number], Actions[step_number]) + '\n')
                self.print_chessboard(States[step_number + 1].Board, f)
                f.write("\n")
            f.write("\n\n")

        print("końcowe wyniki po %d grach: " % (number_of_games))
        print("wygrane białych = %d, wygrane czarnych = %d, remisy = %d" % (num_win_x, num_win_o, num_draws))
        f.write("końcowe wyniki po %d grach: \n" % (number_of_games))
        f.write("wygrane białych = %d, wygrane czarnych = %d, remisy = %d\n" % (num_win_x, num_win_o, num_draws))
        f.close()

    # zapisanie listy akcji i planszy w celach diagnostycznych
    def move_verification(self, State, actions, NextState, player, f):
        player_opo = 3 - player
        if player == 1:
            shift = 0
        else:
            shift = self.BlackShift
        desc_actions = []
        for i in range(len(actions)):
            desc_actions.append(self.action_desc(State, actions[i]))
        f.write("all possible actions: " + str(desc_actions) + "\n")
        king_pos = []
        num_of_rows, num_of_columns = np.shape(State.Board)
        for i in range(num_of_rows):
            for j in range(num_of_columns):
                if State.Board[i, j] == self.King + shift:
                    king_pos = [i, j]
                    break
        if king_pos != []:
            checking_figures = self.list_of_checking_figures(State.Board, king_pos, player_opo)
            for i in range(len(checking_figures)):
                f.write("  checking " + str(num_of_rows - checking_figures[i][0]) + \
                        "," + str(checking_figures[i][1] + 1) + "\n")

        # f.write("\n" + str(step_number) + ": gracz "+str(player)+" wybrał ruch " + \
        #        game_object.action_desc(State,action) + '\n')
        self.print_chessboard(NextState.Board, f)
        f.write("\n")

    # end of class Chess


# -----------------------------------------------------------
# General functions

# x - any vector
# returns probability distribution
def softmax(x):
    max_val = np.max(x)
    return (np.exp(x - max_val) / np.exp(x - max_val).sum())


# string from np.array2string() or simply str() back to array:
def string_to_2Darray(s):
    snum = ""
    li = []
    row = []
    level = -1
    for i in range(len(s)):
        if ((s[i] >= '0') & (s[i] <= '9')) | (s[i] == '.') | (s[i] == '-'):
            snum += s[i]
        else:
            if len(snum) > 0:
                number = int(snum)
                if level == 1:
                    row.append(number)
            snum = ""
            if s[i] == '[':
                level += 1
            elif s[i] == ']':
                if level == 1:
                    li.append(row)
                    # print("row = " + str(row) + " li = " + str(li))
                    row = []
                level -= 1
    # converting into np.array:
    num_of_rows = len(li)
    num_of_columns = 0
    if num_of_rows > 0:
        num_of_columns = len(li[0])
    # print("num_of_rows = " + str(num_of_rows) + " num_of_columns = " + str(num_of_columns))
    if num_of_columns > 0:
        A = np.zeros([num_of_rows, num_of_columns], dtype=int)
        for i in range(num_of_rows):
            for j in range(num_of_columns):
                A[i, j] = li[i][j]
        return A
    else:
        return []

    # end of general functions
    # --------------------------------------------------------------------


def chess_random_games():
    num_of_games = 50

    f = open("chess_random_game.txt", "w")
    # game_object = Chess("szachy_plansza_standardowa.txt")
    # game_object = Chess("szachy_plansza_7x4_bez_kroli.txt")
    game_object = Chess("szachy_plansza_5x5.txt")
    # game_object = Chess("szachy_plansza_3x3.txt")
    # game_object = Chess("szachy_plansza_5x10.txt")
    for game in range(num_of_games):
        f.write("\n\nGra nr " + str(game) + "\n\n")
        State = game_object.initial_state()  # initial state - empty board in tic-tac
        f.write("Początkowa plansza:\n")
        # game_object.print_chessboard_unicode(State.Board,f)
        # game_object.print_chessboard_unicode2(State.Board)
        game_object.print_chessboard(State.Board, f)
        f.write("\n")

        player = 1  # first movement by player 1(cross)

        if_end_of_game = False
        step_number = 0
        Reward_oppo = 0  # opponents reward (after opponent's action)

        while (if_end_of_game == False):  # episode steps loop
            step_number += 1
            Reward = 0
            player_opo = 3 - player
            if player == 1:
                shift = 0
            else:
                shift = game_object.BlackShift
            if not if_end_of_game:
                actions = game_object.actions(State, player)
                num_of_actions = len(actions)
                # action_nr = np.random.randint(num_of_actions)
                action_nr = (step_number + 4) % num_of_actions
                action = actions[action_nr]
                NextState, Reward = game_object.next_state_and_reward(player, State, action)
                desc_actions = []
                for i in range(len(actions)):
                    desc_actions.append(game_object.action_desc(State, actions[i]))
                f.write("all possible actions: " + str(desc_actions) + "\n")
                king_pos = []
                num_of_rows, num_of_columns = np.shape(State.Board)
                for i in range(num_of_rows):
                    for j in range(num_of_columns):
                        if State.Board[i, j] == game_object.King + shift:
                            king_pos = [i, j]
                            break
                if king_pos != []:
                    checking_figures = game_object.list_of_checking_figures(State.Board, king_pos, player_opo)
                    for i in range(len(checking_figures)):
                        f.write("  checking " + str(num_of_rows - checking_figures[i][0]) + \
                                "," + str(checking_figures[i][1] + 1) + "\n")

                f.write("\n" + str(step_number) + ": gracz " + str(player) + " wybrał ruch " + \
                        game_object.action_desc(State, action) + '\n')
                # game_object.print_chessboard_unicode(NextState.Board,f)
                # game_object.print_chessboard_unicode2(NextState.Board)
                game_object.print_chessboard(NextState.Board, f)
                f.write("\n")
            if game_object.end_of_game(Reward, step_number, NextState, action_nr) == True:  # win or draw
                if_end_of_game = True
                f.write("koniec gry, Reward = " + str(Reward) + "\n")
                if game_object.white_objective == game_object.Target_CheckmateKing:
                    if NextState.black_king_mated:
                        f.write("białe dały czarnym mata!" + "\n")
                    elif NextState.black_blocked:
                        f.write("Pat! czarne zostały zablokowane!" + "\n")
                elif game_object.white_objective == game_object.Target_BlockEnemy:
                    if NextState.black_blocked:
                        f.write("Wygrana białych - czarne zostały zablokowane!" + "\n")

                if game_object.black_objective == game_object.Target_CheckmateKing:
                    if NextState.white_king_mated:
                        f.write("czarne dały białym mata!" + "\n")
                    elif NextState.white_blocked:
                        f.write("Pat! białe zostały zablokowane!" + "\n")
                elif game_object.black_objective == game_object.Target_BlockEnemy:
                    if NextState.white_blocked:
                        f.write("Wygrana czarnych - białe zostały zablokowane!" + "\n")

            player = 3 - player
            State = NextState
            f.close()
            f = open("chess_random_game.txt", "a")
    f.close()

# chess_random_games()