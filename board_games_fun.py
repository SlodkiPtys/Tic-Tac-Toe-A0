# functions supporting tictac project 
import numpy as np

class Tictac_general:
    
    def __init__(self, num_of_rows, num_of_columns, num_of_stones, if_adjacent):
        self.num_of_rows = num_of_rows
        self.num_of_columns = num_of_columns
        self.num_of_stones = num_of_stones
        self.if_adjacent = if_adjacent

    def initial_state(self):
        return np.zeros([self.num_of_rows,self.num_of_columns],dtype=int)

    # quicker version of reward which takes into account only
    # pieses in a vertical, horizontel and diagonal / \ sequences
    # contained piece after current move 
    def __reward_after_move(self, A, row, column, player):
        R = 0
        # horizontal direction:
        i = 0
        while True:
            if column + i + 1 < self.num_of_columns:
                if A[row,column + i + 1] == player:
                    i += 1
                else: break
            else: break
        num_of_pieces = i + 1 
        i = 0
        while True:
            if column - i - 1 >= 0:
                if A[row,column - i - 1] == player:
                    i += 1
                else: break
            else: break
        num_of_pieces += i
        if num_of_pieces >= self.num_of_stones:
            R = (player == 1) - (player == 2)

        if R == 0:
            # vertical direction:
            i = 0
            while True:
                if row + i + 1 < self.num_of_rows:
                    if A[row + i + 1,column] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces = i + 1 
            i = 0
            while True:
                if row - i - 1 >= 0:
                    if A[row - i - 1,column] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)

        if R == 0:
            # diagonal \ direction:
            i = 0
            while True:
                if (row + i + 1 < self.num_of_rows)&(column + i + 1 < self.num_of_columns):
                    if A[row + i + 1,column + i + 1] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces = i + 1 
            i = 0
            while True:
                if (row - i - 1 >= 0)&(column - i - 1 >= 0):
                    if A[row - i - 1,column - i - 1] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)

        if R == 0:
            # diagonal / direction:
            i = 0
            while True:
                if (row + i + 1 < self.num_of_rows)&(column - i - 1 >= 0):
                    if A[row + i + 1,column - i - 1] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces = i + 1 
            i = 0
            while True:
                if (row - i - 1 >= 0)&(column + i + 1 < self.num_of_columns):
                    if A[row - i - 1,column + i + 1] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)
        return R

    # checking if end of game (draw)
    def end_of_game(self, _R = 0, _number_of_moves = 0, _Board = [], _action_nr = 0):
        if (np.abs(_R) > 0)|(_number_of_moves >= self.num_of_rows*self.num_of_columns):
            return True
        else: return False


    # output: player, list of states after possible moves, rewards for moves
    def actions(self, A, player = 0):
        actions = []

        empty_cells = np.where(A == 0)
        empty_cells_number = len(empty_cells[0])
        number_of_pieces = self.num_of_rows*self.num_of_columns - empty_cells_number

        if self.if_adjacent:
            if (self.if_adjacent)&(number_of_pieces == 0):
                actions.append([self.num_of_rows//2, self.num_of_columns//2])
            elif (self.if_adjacent)&(number_of_pieces == 1):
                actions.append([self.num_of_rows//2-1, self.num_of_columns//2])
                actions.append([self.num_of_rows//2-1, self.num_of_columns//2-1])
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
                                if (rr >= 0)&(rr < self.num_of_rows)&(cc >= 0)&(cc < self.num_of_columns):
                                    num_of_neibours += (A[rr,cc] != 0)
                    if empty_cells_number == self.num_of_rows*self.num_of_columns:
                        num_of_neibours = 1
                    if (self.if_adjacent == False)|(num_of_neibours > 0):
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
    def print_test_to_file(self, filename,num_win_x, num_win_o, num_draws, Games, Rewards):
        f = open(filename,"w")
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
                        if A[r,c] == 0:
                            row += "_"
                        elif A[r,c] == 1:
                            row += "x"
                        elif A[r,c] == 2:
                            row += "o"
                    row += "  "
                f.write(row + "\n")
            f.write("\n") 

        print("results after %d games: " % (number_of_games))
        print("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.write("results after %d games: " % (number_of_games))
        f.write("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.close()

    def move_verification(self,State,actions,NextState,player,f):
        pass