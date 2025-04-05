# importing the required libraries
import pygame as pg
import numpy as np
import sys
import time
import pdb
from pygame.locals import *


class Interface_Tictactoe:
	def __init__(self, game_object):

		self.width = 400

		# to set height of the game window
		self.height = 400

		# initializing the pygame window
		pg.init()

		# this is used to track time
		self.CLOCK = pg.time.Clock()

		# this method is used to build the
		# infrastructure of the display
		self.screen = pg.display.set_mode((self.width, self.height + 100), 0, 32)

		# setting up a nametag for the
		# game window
		pg.display.set_caption("Tic Tac Toe")

		# loading the images as python object
		initiating_window = pg.image.load("modified_cover.png")
		x_img = pg.image.load("X_modified.png")
		y_img = pg.image.load("o_modified.png")

		# resizing images
		self.initiating_window = pg.transform.scale(
			initiating_window, (self.width, self.height + 100))
		self.x_img = pg.transform.scale(x_img, (80, 80))
		self.o_img = pg.transform.scale(y_img, (80, 80))


	def graphical_board_initiating(self):

		# displaying over the screen
		self.screen.blit(self.initiating_window, (0, 0))

		# updating the display
		pg.display.update()
		#time.sleep(3)
		white = (255, 255, 255)
		self.screen.fill(white)

		line_color = (0, 0, 0)

		# drawing vertical lines
		pg.draw.line(self.screen, line_color, (self.width / 3, 0), (self.width / 3, self.height), 7)
		pg.draw.line(self.screen, line_color, (self.width / 3 * 2, 0),
					(self.width / 3 * 2, self.height), 7)

		# drawing horizontal lines
		pg.draw.line(self.screen, line_color, (0, self.height / 3), (self.width, self.height / 3), 7)
		pg.draw.line(self.screen, line_color, (0, self.height / 3 * 2),
					(self.width, self.height / 3 * 2), 7)



	def draw_status(self,player,winner,extra_text=""):
		if winner is None:
			if player == 1:
				message = "x's Turn"
			else:
				message = "o's Turn"
		else:
			if winner == 1:
				message = "x won !"
			elif winner == 2:
				message = "o won !"
			elif winner == 0:
				message = "Game Draw !"
		print(message)

		# setting a font object
				# setting a font object
		font = pg.font.Font(None, 20)

		# copy the rendered message onto the board
		# creating a small block at the bottom of the main display
		self.screen.fill((50, 50, 50), (0, self.height, self.width, 100))

		# setting the font properties like
		# color and width of the text
		text = font.render(message, 1, (255, 255, 255))
		text_rect = text.get_rect(center=(self.width / 2, self.height+100-70))
		self.screen.blit(text, text_rect)

		text2 = font.render(extra_text, 1, (255, 255, 255))
		text2_rect = text2.get_rect(center=(self.width/2, self.height+100-40))
		self.screen.blit(text2, text2_rect)
		pg.display.update()

	def draw_end_of_game(self, State,player,winner):
		# checking for winning rows
		for row in range(3):
			if((State[row][0] == State[row][1] == State[row][2]) and (State[row][0] > 0)):
				pg.draw.line(self.screen, (250, 0, 0),
							(0, (row + 1)*self.height / 3 - self.height / 6),
							(self.width, (row + 1)*self.height / 3 - self.height / 6), 4)
				break
		# checking for winning columns
		for col in range(3):
			if((State[0][col] == State[1][col] == State[2][col]) and (State[0][col] > 0)):
				pg.draw.line(self.screen, (250, 0, 0), ((col + 1) * self.width / 3 - self.width / 6, 0),
							((col + 1) * self.width / 3 - self.width / 6, self.height), 4)
				break
		# check for diagonal winners
		if (State[0][0] == State[1][1] == State[2][2]) and (State[0][0] > 0):
			# game won diagonally left to right
			pg.draw.line(self.screen, (250, 70, 70), (50, 50), (350, 350), 4)

		if (State[0][2] == State[1][1] == State[2][0]) and (State[0][2] > 0):
			# game won diagonally right to left
			pg.draw.line(self.screen, (250, 70, 70), (350, 50), (50, 350), 4)

		#draw_status(player,winner)


	def drawXO(self,row, col, player):
		# for the first row, the image
		# should be pasted at a x coordinate
		# of 30 from the left margin
		if row == 0:
			posx = 30

		# for the second row, the image
		# should be pasted at a x coordinate
		# of 30 from the game line
		if row == 1:
			# margin or width / 3 + 30 from
			# the left margin of the window
			posx = self.width / 3 + 30

		if row == 2:
			posx = self.width / 3 * 2 + 30

		if col == 0:
			posy = 30

		if col == 1:
			posy = self.height / 3 + 30

		if col == 2:
			posy = self.height / 3 * 2 + 30
		# setting up the required board
		# value to display

		if player == 1:

			# pasting x_img over the screen
			# at a coordinate position of
			# (pos_y, posx) defined in the
			# above code
			self.screen.blit(self.x_img, (posy, posx))
		else:
			self.screen.blit(self.o_img, (posy, posx))
		pg.display.update()


	def user_click(self):
		# get coordinates of mouse click
		x, y = pg.mouse.get_pos()
		
		# get column of mouse click (1-3)
		if(x < self.width / 3):
			col = 0
		elif (x < self.width / 3 * 2):
			col = 1
		elif(x < self.width):
			col = 2
		else:
			col = None

		# get row of mouse click (1-3)
		if(y < self.height / 3):
			row = 0
		elif (y < self.height / 3 * 2):
			row = 1
		elif(y < self.height):
			row = 2
		else:
			row = None
			
		return row, col

	# play game with strategy and player using this strategy e.g. 
	# player 1 playing by x in tic-tac-toe:
	def play_with_strategy(self,game_object, strategy, str_player):
		human_player = 3-str_player	
		end_of_interaction = False

		while end_of_interaction == False:   # loop for games

			player = 1                                        # player which starts the game
			winner = None								      # player who won the game
			State = game_object.initial_state()               # initial state - empty board in tic-tac
			end_of_game = False
			step_number = 0
			self.graphical_board_initiating()
			self.draw_status(player,None)

			while end_of_game == False:      # loop for moves
				step_number += 1
				actions = game_object.actions(State, player)  # all possible actions in state State

				if player == human_player:                    # human move
					legal_action_was_choosen = False
					while (legal_action_was_choosen == False)&(end_of_game == False):  # try as long as legal move was chosen
						row = col = None                      # move coordinates
						for event in pg.event.get():
							if event.type == QUIT:
								end_of_interaction = True
								pg.quit()
								sys.exit()
							elif event.type == MOUSEBUTTONDOWN:  
								#pdb.set_trace()
								row,col = self.user_click()
								print("click: row = "+str(row)+ ", column = "+str(col))
								#pdb.set_trace()
								#if winner or draw:
								#	reset_game()
							elif event.type == KEYDOWN:
								if (event.key == K_BACKSPACE)|(event.key == K_END):
									end_of_game = True
								elif event.key == K_SPACE:
									row,col = self.user_click()
									#if winner or draw:
									#	reset_game()
						if (row != None) and (col != None) and ([row,col] in actions):
							legal_action_was_choosen = True
							print("legal move!")
						
					for i in range(len(actions)):
						if [row,col] == actions[i]:
							action_nr = i
				else:
					action_nr, _ = strategy.choose_action(State,player)
					if action_nr == None:
						print("action_nr = "+str(action_nr)+" State = "+str(State)+ " player = "+str(player))
						action_nr = np.random.randint(len(actions))
					

				print("action_nr = " + str(action_nr)+" actions = "+str(actions))
				NextState, Reward =  game_object.next_state_and_reward(player,State, actions[action_nr])
				self.drawXO(actions[action_nr][0], actions[action_nr][1], player)
				

				if game_object.end_of_game(Reward,step_number,State,action_nr):      # win or draw
					end_of_game = True
					if Reward == 1:
						winner = 1
					elif Reward == -1:
						winner = 2
					else:
						winner = 0
					self.draw_end_of_game(NextState,player,winner)
					
				self.draw_status(3-player,winner," End - new game")

				if end_of_game:
					time.sleep(3)
					
				pg.display.update()
				fps = 30
				self.CLOCK.tick(fps)

				player = 3 - player
				State = NextState



# ----------------------------------------------------
class Interface_Tictac_general(Interface_Tictactoe):
	def __init__(self,game_object):

		self.num_of_rows = game_object.num_of_rows
		self.num_of_columns = game_object.num_of_columns
		self.num_of_stones = game_object.num_of_stones
		self.if_adjacent = game_object.if_adjacent

		self.width = 100*self.num_of_columns

		# to set height of the game window
		self.height = 100*self.num_of_rows

		if max(self.width, self.height) > 600:
			new_width = int(self.width * 600/max(self.width, self.height))
			new_height = int(self.height * 600/max(self.width, self.height))
			self.width = new_width
			self.height = new_height

		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows

		# initializing the pygame window
		pg.init()

		# this is used to track time
		self.CLOCK = pg.time.Clock()

		# this method is used to build the
		# infrastructure of the display
		self.screen = pg.display.set_mode((self.width, self.height + 100), 0, 32)

		# setting up a nametag for the
		# game window
		pg.display.set_caption("Tic Tac General")

		# loading the images as python object
		initiating_window = pg.image.load("modified_cover.png")
		x_img = pg.image.load("X_modified.png")
		y_img = pg.image.load("o_modified.png")

		# resizing images
		self.initiating_window = pg.transform.scale(
			initiating_window, (self.width, self.height + 100))
		self.x_img = pg.transform.scale(x_img, (int(d_col*4/5), int(d_row*4/5)))
		self.o_img = pg.transform.scale(y_img, (int(d_col*4/5), int(d_row*4/5)))


	def graphical_board_initiating(self):

		# displaying over the screen
		self.screen.blit(self.initiating_window, (0, 0))

		# updating the display
		pg.display.update()
		#time.sleep(3)
		white = (255, 255, 255)
		self.screen.fill(white)

		line_color = (0, 0, 0)

		# drawing vertical lines
		for col in range(self.num_of_columns-1):
			pg.draw.line(self.screen, line_color, (self.width / self.num_of_columns * (col+1), 0),
					(self.width / self.num_of_columns * (col+1), self.height), 7)

		# drawing horizontal lines
		for row in range(self.num_of_rows-1):
			pg.draw.line(self.screen, line_color, (0, self.height / self.num_of_rows * (row+1)),
					(self.width, self.height / self.num_of_rows * (row+1)), 7)

	def draw_status(self,player,winner,extra_text=""):
		if winner is None:
			if player == 1:
				message = "x's Turn"
			else:
				message = "o's Turn"
		else:
			if winner == 1:
				message = "x won !"
			elif winner == 2:
				message = "o won !"
			elif winner == 0:
				message = "Game Draw !"
		print(message)

		# setting a font object
				# setting a font object
		font = pg.font.Font(None, 20)

		# copy the rendered message onto the board
		# creating a small block at the bottom of the main display
		self.screen.fill((50, 50, 50), (0, self.height, self.width, 100))

		# setting the font properties like
		# color and width of the text
		text = font.render(message, 1, (255, 255, 255))
		text_rect = text.get_rect(center=(self.width / 2, self.height+100-70))
		self.screen.blit(text, text_rect)

		text2 = font.render(extra_text, 1, (255, 255, 255))
		text2_rect = text2.get_rect(center=(self.width/2, self.height+100-40))
		self.screen.blit(text2, text2_rect)
		pg.display.update()


	def draw_end_of_game(self, State,player,winner):
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows

		# checking for winning in horizontal direction:
		for row in range(self.num_of_rows):
			x_seq_start = -1
			o_seq_start = -1
			for col in range(self.num_of_columns):
				if State[row][col] == 1:
					if x_seq_start == -1:
						x_seq_start = col
				else:
					x_seq_start = -1

				if State[row][col] == 2:
					if o_seq_start == -1:
						o_seq_start = col
				else:
					o_seq_start = -1

				if (x_seq_start > -1)&(col - x_seq_start + 1 == self.num_of_stones): 
					x1 = int(x_seq_start*d_col)
					y1 = int((row + 1)*d_row  - d_row/2)
					x2 = int((col+1)*d_col)
					y2 = int((row + 1)*d_row - d_row/2)
					pg.draw.line(self.screen, (250, 0, 0), (x1,y1), (x2 ,y2 ), 8)
					print("x_seq_start = "+str(x_seq_start)+" o_seq_start = "+str(o_seq_start)+" col = "+str(col))
					print("d_row = "+str(d_row)+" d_col = "+str(d_col)," x1 = "+str(x1)+" y1 = "+str(y1)+" x2 = "+str(x2)+" y2 = "+str(y2))
					break
				if (o_seq_start > -1)&(col - o_seq_start + 1 == self.num_of_stones):
					x1 = int(o_seq_start*d_col)
					y1 = int((row + 1)*d_row  - d_row/2)
					x2 = int((col+1)*d_col)
					y2 = int((row + 1)*d_row - d_row/2)
					pg.draw.line(self.screen, (250, 0, 0), (x1,y1), (x2 ,y2 ), 8)
					break

		#checking for winning in vertical direction:
		for col in range(self.num_of_columns):		
			x_seq_start = -1
			o_seq_start = -1
			for row in range(self.num_of_rows):
				if State[row][col] == 1:
					if x_seq_start == -1:
						x_seq_start = row
				else:
					x_seq_start = -1

				if State[row][col] == 2:
					if o_seq_start == -1:
						o_seq_start = row
				else:
					o_seq_start = -1

				if (x_seq_start > -1)&(row - x_seq_start + 1 == self.num_of_stones):
					x1 = int((col + 1) * d_col - d_col/2)
					y1 = int(x_seq_start*d_row)
					x2 = int((col + 1) * d_col - d_col/2)
					y2 = int((row+1)*d_row)
					pg.draw.line(self.screen, (250, 0, 0), (x1,y1), (x2 ,y2 ), 8)
					break
				if (o_seq_start > -1)&(row - o_seq_start + 1 == self.num_of_stones):
					x1 = int((col + 1) * d_col - d_col/2)
					y1 = int(o_seq_start*d_row)
					x2 = int((col + 1) * d_col - d_col/2)
					y2 = int((row+1)*d_row)
					pg.draw.line(self.screen, (250, 0, 0), (x1,y1), (x2 ,y2 ), 8)
					break
		
		# checking for winning in \ direction:
		start_squares = []
		for row in range(self.num_of_rows-1):
			start_squares.append([self.num_of_rows-row-1,0])
		for col in range(self.num_of_columns):
			start_squares.append([0,col])

		for st in range(len(start_squares)):
			row = start_squares[st][0]
			col = start_squares[st][1]
			d_start = col - row
			x_seq_start = -1
			o_seq_start = -1
			while (row < self.num_of_rows)&(col < self.num_of_columns):
				if State[row][col] == 1:
					if x_seq_start == -1:
						x_seq_start = row
				else:
					x_seq_start = -1

				if State[row][col] == 2:
					if o_seq_start == -1:
						o_seq_start = row
				else:
					o_seq_start = -1

				if (x_seq_start > -1)&(row - x_seq_start + 1 == self.num_of_stones):
					x1 = int((x_seq_start+d_start) * d_col)
					y1 = int(x_seq_start*d_row)
					x2 = int((row+1+d_start) * d_col)
					y2 = int((row+1)*d_row)
					pg.draw.line(self.screen, (250, 0, 0), (x1,y1), (x2 ,y2 ), 8)
					break
				if (o_seq_start > -1)&(row - o_seq_start + 1 == self.num_of_stones):
					x1 = int((o_seq_start+d_start) * d_col)
					y1 = int(o_seq_start*d_row)
					x2 = int((row+1+d_start) * d_col)
					y2 = int((row+1)*d_row)
					pg.draw.line(self.screen, (250, 0, 0), (x1,y1), (x2 ,y2 ), 8)
					break

				row += 1
				col += 1 


		# checking for winning in / direction:
		start_squares = []
		for col in range(self.num_of_columns):
			start_squares.append([0,col])
		for row in range(self.num_of_rows-1):
			start_squares.append([row+1,self.num_of_columns-1])

		for st in range(len(start_squares)):
			row = start_squares[st][0]
			col = start_squares[st][1]
			#pdb.set_trace()
			d_start = col - row
			x_seq_start = -1
			o_seq_start = -1
			while (row < self.num_of_rows)&(col >= 0):
				if State[row][col] == 1:
					if x_seq_start == -1:
						x_seq_start = col
				else:
					x_seq_start = -1

				if State[row][col] == 2:
					if o_seq_start == -1:
						o_seq_start = row
				else:
					o_seq_start = -1

				if (x_seq_start > -1)&(x_seq_start - col + 1 == self.num_of_stones):
					x1 = int((col + self.num_of_stones)*d_col)
					y1 = int((row+1-self.num_of_stones)*d_row)
					x2 = int((col) * d_col)
					y2 = int((row+1)*d_row)
					pg.draw.line(self.screen, (250, 0, 0), (x1,y1), (x2 ,y2 ), 8)
					break
				if (o_seq_start > -1)&(row - o_seq_start + 1 == self.num_of_stones):
					x1 = int((start_squares[st][0]-o_seq_start+d_start) * d_col)
					y1 = int(o_seq_start*d_row)
					x2 = int((col) * d_col)
					y2 = int((row+1)*d_row)
					pg.draw.line(self.screen, (250, 0, 0), (x1,y1), (x2 ,y2 ), 8)
					break

				row += 1
				col -= 1 


	def drawXO(self,row, col, player):
		# for the first row, the image
		# should be pasted at a x coordinate
		# of 30 from the left margin
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows

		# for the second row, the image
		# should be pasted at a x coordinate
		# of 30 from the game line

		posy = d_col * col + int(24*3/self.num_of_columns)
		posx = d_row * row + int(24*3/self.num_of_rows)

		#print("posx = "+str(posx)+" posy = "+str(posy))

		# setting up the required board
		# value to display

		if player == 1:
			self.screen.blit(self.x_img, (posy, posx))
		else:
			self.screen.blit(self.o_img, (posy, posx))
		pg.display.update()


	def user_click(self):
		# get coordinates of mouse click
		x, y = pg.mouse.get_pos()
		
		# get column of mouse click (1-3)
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows
		col = int(x / d_col)
		row = int(y / d_row)
		if (col < 0)|(col > self.num_of_columns-1):
			col = None
		if (row < 0)|(row > self.num_of_rows-1):
			row = None
			
		return row, col



# --------------------------------------------------------
class Interface_Connect4(Interface_Tictac_general):
	def __init__(self,game_object):

		self.num_of_rows = game_object.num_of_rows
		self.num_of_columns = game_object.num_of_columns
		self.num_of_stones = 4
		self.if_adjacent = True

		self.width = 100*self.num_of_columns

		# to set height of the game window
		self.height = 100*self.num_of_rows

		if max(self.width, self.height) > 1000:
			self.width = int(self.width * 1000/max(self.width, self.height))
			self.height = int(self.height * 1000/max(self.width, self.height))

		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows

		# initializing the pygame window
		pg.init()

		# this is used to track time
		self.CLOCK = pg.time.Clock()

		# this method is used to build the
		# infrastructure of the display
		self.screen = pg.display.set_mode((self.width, self.height + 100), 0, 32)

		# setting up a nametag for the
		# game window
		pg.display.set_caption("Connect4")

		# loading the images as python object
		initiating_window = pg.image.load("modified_cover.png")
		x_img = pg.image.load("X_modified.png")
		y_img = pg.image.load("o_modified.png")

		# resizing images
		self.initiating_window = pg.transform.scale(
			initiating_window, (self.width, self.height + 100))
		self.x_img = pg.transform.scale(x_img, (int(d_col*4/5), int(d_row*4/5)))
		self.o_img = pg.transform.scale(y_img, (int(d_col*4/5), int(d_row*4/5)))


	# play game with strategy and player using this strategy e.g. 
	# player 1 playing by x in tic-tac-toe:
	def play_with_strategy(self,game_object, strategy, str_player):
		human_player = 3-str_player	
		end_of_interaction = False

		while end_of_interaction == False:   # loop for games

			player = 1                                        # player which starts the game
			winner = None								      # player who won the game
			State = game_object.initial_state()               # initial state - empty board in tic-tac
			end_of_game = False
			step_number = 0
			self.graphical_board_initiating()
			self.draw_status(player,None)

			while end_of_game == False:      # loop for moves
				step_number += 1
				actions = game_object.actions(State, player)  # all possible actions in state State
				num_of_actions = len(actions) 

				if player == human_player:                    # human move
					legal_action_was_choosen = False
					while legal_action_was_choosen == False:  # try as long as legal move was chosen
						row = col = None                      # move coordinates
						for event in pg.event.get():
							if event.type == QUIT:
								end_of_interaction = True
								pg.quit()
								sys.exit()
							elif event.type == MOUSEBUTTONDOWN:  
								#pdb.set_trace()
								row,col = self.user_click()
								print("click: row = "+str(row)+ ", column = "+str(col))
								#pdb.set_trace()
								#if winner or draw:
								#	reset_game()
							if event.type == KEYDOWN:
								if event.key == K_BACKSPACE:
									end_of_game = True
								if event.key == K_SPACE:
									row,col = self.user_click()
									#if winner or draw:
									#	reset_game()
						if (row != None) and (col != None) and ([row,col] in actions):
							legal_action_was_choosen = True
							print("legal move!")
						
					for i in range(len(actions)):
						if [row,col] == actions[i]:
							action_nr = i
				else:
					action_nr, _ = strategy.choose_action(State,player)
					if action_nr == None:
						print("action_nr = "+str(action_nr)+" State = "+str(State)+ " player = "+str(player))
						action_nr = np.random.randint(len(actions))
					

				print("action_nr = " + str(action_nr)+" actions = "+str(actions))
				NextState, Reward =  game_object.next_state_and_reward(player,State, actions[action_nr])
				self.drawXO(actions[action_nr][0], actions[action_nr][1], player)
				

				if game_object.end_of_game(Reward,step_number,State,action_nr):      # win or draw
					end_of_game = True
					if Reward == 1:
						winner = 1
					elif Reward == -1:
						winner = 2
					else:
						winner = 0
					self.draw_end_of_game(NextState,player,winner)
					
				self.draw_status(player,winner)

				if end_of_game:
					time.sleep(3)
					
				pg.display.update()
				fps = 30
				self.CLOCK.tick(fps)

				player = 3 - player
				State = NextState


# ------------------------------
# ------------------------------
# ------------------------------
class Interface_Chess():
	def __init__(self,game_object):

		self.num_of_rows = game_object.num_of_rows
		self.num_of_columns = game_object.num_of_columns
		self.game_object = game_object

		self.width = 100*self.num_of_columns
		self.height = 100*self.num_of_rows

		if max(self.width, self.height) > 600:
			new_width = int(self.width * 600/max(self.width, self.height))
			new_height = int(self.height * 600/max(self.width, self.height))
			self.width = new_width
			self.height = new_height

		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows

		# initializing the pygame window
		pg.init()

		# this is used to track time
		self.CLOCK = pg.time.Clock()

		# this method is used to build the
		# infrastructure of the display
		self.screen = pg.display.set_mode((self.width, self.height + 100), 0, 32)

		# setting up a nametag for the
		# game window
		pg.display.set_caption("Chess")

		# loading the images as python object
		initiating_window = pg.image.load("modified_cover.png")
		x_img = pg.image.load("X_modified.png")
		y_img = pg.image.load("o_modified.png")

		# resizing images
		self.initiating_window = pg.transform.scale(
			initiating_window, (self.width, self.height + 100))
		self.x_img = pg.transform.scale(x_img, (int(d_col*4/5), int(d_row*4/5)))
		self.o_img = pg.transform.scale(y_img, (int(d_col*4/5), int(d_row*4/5)))
		self.board_color = (140,90,0)

	def graphical_board_initiating(self):

		# displaying over the screen
		self.screen.blit(self.initiating_window, (0, 0))

		
		#time.sleep(3)
		white = (255, 255, 255)
		self.screen.fill(white)

		line_color = (0, 0, 0)

		# # drawing vertical lines
		# for col in range(self.num_of_columns-1):
		# 	pg.draw.line(self.screen, line_color, (self.width / self.num_of_columns * (col+1), 0),
		# 			(self.width / self.num_of_columns * (col+1), self.height), 7)

		# # drawing horizontal lines
		# for row in range(self.num_of_rows-1):
		# 	pg.draw.line(self.screen, line_color, (0, self.height / self.num_of_rows * (row+1)),
		# 			(self.width, self.height / self.num_of_rows * (row+1)), 7)
			
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows

		# drawing chessboard:
		for row in range(self.num_of_rows):
			for col in range(self.num_of_columns):				
				if ((row + col)%2) == 1:
					#pg.draw.rect(self.screen, (100, 75, 0), [int(row*d_row), int(col*d_col), int(d_row), int(d_col)])
					pg.draw.rect(self.screen, self.board_color, [int(col*d_col),int(row*d_row), int(d_col),  int(d_row)])
					#self.screen.fill((100, 75, 0), (int(row*d_row), int(col*d_col), int(d_row), int(d_col)))
		# drawing pieces:
		for row in range(self.num_of_rows):
			for col in range(self.num_of_columns):
				self.draw_piece(row,col,self.game_object.InitialBoard[row,col])	

		col_dict = {0:'a', 1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m', \
                    13:'n',14:'o',15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',\
                    25:'z'}
		
		# updating the display
		pg.display.update()


	def draw_status(self,player,winner,extra_text = ""):
		if winner is None:
			if player == 2:
				message = "black's Turn"
			else:
				message = "white's Turn"
		else:
			if winner == 1:
				message = "white won !"
			elif winner == 2:
				message = "black won !"
			elif winner == 0:
				message = "Game Draw !"
		print(message)

		# setting a font object
		font = pg.font.Font(None, 20)

		# copy the rendered message onto the board
		# creating a small block at the bottom of the main display
		self.screen.fill((50, 50, 50), (0, self.height, self.width, 100))

		# setting the font properties like
		# color and width of the text
		text = font.render(message, 1, (255, 255, 255))
		text_rect = text.get_rect(center=(self.width / 2, self.height+100-70))
		self.screen.blit(text, text_rect)

		text2 = font.render(extra_text, 1, (255, 255, 255))
		text2_rect = text2.get_rect(center=(self.width/2, self.height+100-40))
		self.screen.blit(text2, text2_rect)
		pg.display.update()


	def draw_end_of_game(self, State,player,winner):
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows

	def draw_frame(self,row,col):
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows
		pg.draw.rect(self.screen, (255,   0, 0),
                 [int(col*d_col), int(row*d_row), int(d_col), int(d_row)], 4)
		pg.display.update()
		
	def draw_empty(self,row,col):
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows
		if ((row+col)%2) == 1:
			color = self.board_color
		else:
			color = (255,255,255)
		pg.draw.rect(self.screen, color, [int(col*d_col), int(row*d_row), int(d_col), int(d_row)])
		pg.display.update()

	def draw_piece(self, row, col, piece, red_contour = False):
		# for the first row, the image
		# should be pasted at a x coordinate
		# of 30 from the left margin
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows
		
		posy = d_col * col 
		posx = d_row * row 

		if red_contour:
			contour_color = (255,0,0)
			line_thickness = 4
		else:
			if piece < 1000:
				contour_color = (0,0,0)
			else:
				contour_color = (255,255,255)
			line_thickness = 2

		# if player == 1:
		# 	self.screen.blit(self.x_img, (posy, posx))			
		# else:
		# 	self.screen.blit(self.o_img, (posy, posx))

		# Pawn = 1         # static variables (belonging to class)
		# Knight = 2       # white pieces identifies
		# Bishop = 3
		# Rook = 4
		# Queen = 5
		# King = 6
		
		# BlackShift = 1000   # shift of black pieces identifies

		if piece == self.game_object.Pawn:
			points = [[3, 7], [3, 8], [7,8], [7,7],[6,7],[6,5],[4,5],[4,7],[3,7]]
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)

			pg.draw.circle(self.screen,(255, 255, 255),np.array([posy,posx]) + np.array([5, 3.5])*np.array([d_row,d_col])/10, 
				  1.5*d_row/10)
			pg.draw.circle(self.screen,contour_color,np.array([posy,posx]) + np.array([5, 3.5])*np.array([d_row,d_col])/10, 
				  1.5*d_row/10,line_thickness)
		elif piece == self.game_object.Pawn + self.game_object.BlackShift:
			points = [[3, 7], [3, 8], [7,8], [7,7],[6,7],[6,5],[4,5],[4,7],[3,7]]
			pg.draw.polygon(self.screen, (0, 0, 0), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.circle(self.screen,(0,0,0),np.array([posy,posx]) + np.array([5, 3.5])*np.array([d_row,d_col])/10, 
				  1.5*d_row/10)
			pg.draw.circle(self.screen,contour_color,np.array([posy,posx]) + np.array([5, 3.5])*np.array([d_row,d_col])/10, 
				  1.5*d_row/10,line_thickness)
		elif piece == self.game_object.Knight:
			points = [[3,9],[7,9],[7,8],[6,4],[4,2],[4,3],[2,5],[3,6],[4,5.5],[4,6.5],[3,8]]
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.circle(self.screen,(0,0,0),np.array([posy,posx]) + np.array([4, 4])*np.array([d_row,d_col])/10, 
				  0.5*d_row/10)
		elif piece == self.game_object.Knight + self.game_object.BlackShift:
			points = [[3,9],[7,9],[7,8],[6,4],[4,2],[4,3],[2,5],[3,6],[4,5.5],[4,6.5],[3,8]]
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.circle(self.screen,(255, 255, 255),np.array([posy,posx]) + np.array([4, 4])*np.array([d_row,d_col])/10, 
				  0.5*d_row/10)
		elif piece == self.game_object.Bishop:
			points = [[3, 8],[3,9],[7,9],[7,8],[6,8],[6,5],[7,4],[6, 2.5],[4, 2.5],[3,4],[4,5],[4,8]]
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
			points2 = [[6, 2.5],[5,1], [4, 2.5]]
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points2)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, (255, 255, 255), 
				   np.array([posy,posx]) + np.array(points2)*np.array([d_row,d_col])/10,line_thickness)

		elif piece == self.game_object.Bishop + self.game_object.BlackShift:
			points = [[3, 8],[3,9],[7,9],[7,8],[6,8],[6,5],[7,4],[6, 2.5],[4, 2.5],[3,4],[4,5],[4,8]]
			#pdb.set_trace()
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
			points2 = [[6, 2.5],[5,1], [4, 2.5]]
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points2)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, (0,0,0), 
				   np.array([posy,posx]) + np.array(points2)*np.array([d_row,d_col])/10,line_thickness)
		elif piece == self.game_object.Rook:
			points = [[3, 8], [3, 9], [7,9], [7,8],[6,8],[6, 5.5],[7.5,5],[7.5,3],[6.5,3],[6.5,4],[5.5,4],[5.5,3],[4.5,3],\
				[4.5,4],[3.5,4],[3.5,3],[2.5,3],[2.5,5],[4, 5.5],[4,8]]
			pg.draw.polygon(self.screen, (255, 255, 255), 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
		elif piece == self.game_object.Rook + self.game_object.BlackShift:
			points = [[3, 8], [3, 9], [7,9], [7,8],[6,8],[6, 5.5],[7.5,5],[7.5,3],[6.5,3],[6.5,4],[5.5,4],[5.5,3],[4.5,3],\
				[4.5,4],[3.5,4],[3.5,3],[2.5,3],[2.5,5],[4, 5.5],[4,8]]
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
		elif piece == self.game_object.Queen:
			points = [[2.5,8],[2.5,9],[7.5,9],[7.5,8],[6.5,8],[6.5,6],[8,3],[6,4],[5,2],[4,4],[2,3],[3.5,6],[3.5,8]]
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
		elif piece == self.game_object.Queen + self.game_object.BlackShift:
			points = [[2.5,8],[2.5,9],[7.5,9],[7.5,8],[6.5,8],[6.5,6],[8,3],[6,4],[5,2],[4,4],[2,3],[3.5,6],[3.5,8]]
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, 
				   np.array([posy,posx]) + np.array(points)*np.array([d_row,d_col])/10,line_thickness)
		elif piece == self.game_object.King:
			points1 = [[2,8],[2,9],[8,9],[8,8],[5,7]]
			points2 = [[2.5,6.5],[3,7.5],[5,7],[5,5],[4,4],[3,4]]
			points3 = [[5,5],[5,7],[7,7.5],[7.5,6.5],[7,4],[6,4]]
			points4 = [[4,4],[5,5],[6,4],[5,3]]
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points1)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points2)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points3)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, (255, 255, 255), np.array([posy,posx]) + np.array(points4)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, np.array([posy,posx]) + np.array(points1)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.polygon(self.screen, contour_color, np.array([posy,posx]) + np.array(points2)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.polygon(self.screen, contour_color, np.array([posy,posx]) + np.array(points3)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.polygon(self.screen, contour_color, np.array([posy,posx]) + np.array(points4)*np.array([d_row,d_col])/10,line_thickness)
		elif piece == self.game_object.King + self.game_object.BlackShift:
			points1 = [[2,8],[2,9],[8,9],[8,8],[5,7]]
			points2 = [[2.5,6.5],[3,7.5],[5,7],[5,5],[4,4],[3,4]]
			points3 = [[5,5],[5,7],[7,7.5],[7.5,6.5],[7,4],[6,4]]
			points4 = [[4,4],[5,5],[6,4],[5,3]]
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points1)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points2)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points3)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, (0,0,0), np.array([posy,posx]) + np.array(points4)*np.array([d_row,d_col])/10)
			pg.draw.polygon(self.screen, contour_color, np.array([posy,posx]) + np.array(points1)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.polygon(self.screen, contour_color, np.array([posy,posx]) + np.array(points2)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.polygon(self.screen, contour_color, np.array([posy,posx]) + np.array(points3)*np.array([d_row,d_col])/10,line_thickness)
			pg.draw.polygon(self.screen, contour_color, np.array([posy,posx]) + np.array(points4)*np.array([d_row,d_col])/10,line_thickness)
		pg.display.update()


	def user_click(self):
		# get coordinates of mouse click
		x, y = pg.mouse.get_pos()
		
		# get column of mouse click (1-3)
		d_col = self.width / self.num_of_columns   # width of single column
		d_row = self.height / self.num_of_rows
		col = int(x / d_col)
		row = int(y / d_row)
		if (col < 0)|(col > self.num_of_columns-1):
			col = None
		if (row < 0)|(row > self.num_of_rows-1):
			row = None
			
		return row, col

	# play game with strategy and player using this strategy e.g. 
	# player 1 playing by x in tic-tac-toe:
	def play_with_strategy(self,game_object, strategy, str_player):
		human_player = 3-str_player	
		end_of_interaction = False

		while end_of_interaction == False:   # loop for games

			player = 1                                        # player which starts the game
			winner = None								      # player who won the game
			State = game_object.initial_state()               # initial state - empty board in tic-tac
			end_of_game = False
			step_number = 0
			self.graphical_board_initiating()
			self.draw_status(player,None)
			print("New game!")

			while end_of_game == False:      # loop for moves
				step_number += 1
				actions = game_object.actions(State, player)  # all possible actions in state State
				actions_point_from = []
				actions_point_fromto = []			
				choose_promotion = False
				promotion_piece = None

				for i in range(len(actions)):
					actions_point_from.append([actions[i][1],actions[i][2]])
					actions_point_fromto.append([actions[i][1],actions[i][2],actions[i][3],actions[i][4]])
				num_of_actions = len(actions) 

				if player == human_player:                    # human move
					# human should choose starting point and destination point of his piece:
					legal_starting_point_was_choosen = False
					legal_action_was_choosen = False
					fromto = []
					starting_point = []
					while (legal_action_was_choosen == False)&(end_of_game == False):  # try as long as legal move was chosen
						row = col = None                      # move coordinates
						for event in pg.event.get():
							if event.type == QUIT:
								end_of_interaction = True
								pg.quit()
								sys.exit()
							elif event.type == MOUSEBUTTONDOWN:  
								#pdb.set_trace()
								row,col = self.user_click()
								print("click: row = "+str(row)+ ", column = "+str(col))
								#pdb.set_trace()
								#if winner or draw:
								#	reset_game()
							elif event.type == KEYDOWN:
								if (event.key == K_BACKSPACE)|(event.key == K_END):
									end_of_game = True
									legal_starting_point_was_choosen = False
									legal_action_was_choosen = False
									starting_point = None
									fromto = None
								elif event.key == K_ESCAPE:
									legal_starting_point_was_choosen = False
									legal_action_was_choosen = False
									if starting_point != None:
										[__row,__col] = starting_point
										self.draw_piece(__row, __col, State.Board[__row,__col])
									starting_point = None
									fromto = None
									
								elif event.key == K_SPACE:
									row,col = self.user_click()
								elif event.key == pg.K_q:				
									promotion_piece = self.game_object.Queen
									if player == 2: promotion_piece += self.game_object.Blackshift
									row = fromto[2]
									col = fromto[3]
									choose_promotion = False
								elif event.key == pg.K_k:					
									promotion_piece = self.game_object.Knight
									if player == 2: promotion_piece += self.game_object.Blackshift
									row = fromto[2]
									col = fromto[3]
									choose_promotion = False
								elif event.key == pg.K_r:					
									promotion_piece = self.game_object.Rook
									if player == 2: promotion_piece += self.game_object.Blackshift
									row = fromto[2]
									col = fromto[3]
									choose_promotion = False
								elif event.key == pg.K_b:					
									promotion_piece = self.game_object.Bishop
									if player == 2: promotion_piece += self.game_object.Blackshift
									row = fromto[2]
									col = fromto[3]
									choose_promotion = False								

						if legal_starting_point_was_choosen == False:
							if (row != None) and (col != None) and ([row,col] in actions_point_from):
								legal_starting_point_was_choosen = True
								starting_point = [row,col] 
								self.draw_piece(row, col, State.Board[row,col],red_contour=True)
								self.draw_status(player,winner,"choose target place, ESC - choose starting place again!")
								#self.draw_frame(row,col)
								print("legal starting point!")
						elif choose_promotion:
							pass
						else:
							if (row != None) and (col != None) and \
								([starting_point[0],starting_point[1],row,col] in actions_point_fromto):
								fromto = [starting_point[0],starting_point[1],row,col] 
								for i in range(len(actions)):
									if (fromto == [actions[i][1],actions[i][2],actions[i][3],actions[i][4]]):
										action_nr = i
								if len(actions[action_nr]) > 5:
									if promotion_piece == None:
										choose_promotion = True
										self.draw_status(player,winner,\
						   					"choose a promotional piece: Q-queen, K-knight, R-rook, B-bishop")
										print("choose a promotional piece: Q-queen, K-knight, R-rook, B-bishop!")
									else:
										choose_promotion = False
										for i in range(len(actions)):
											if (fromto == [actions[i][1],actions[i][2],actions[i][3],actions[i][4]])&\
												(len(actions[i]) > 5):
												if promotion_piece == actions[i][5]:
													action_nr = i
										legal_action_was_choosen = True
										print("move complete with promotion!")			
								else:
									legal_action_was_choosen = True
									print("move complete! End - new game.")				 
					
				else:  # strategy move
					action_nr, _ = strategy.choose_action(State,player)
					if action_nr == None:
						print("action_nr = "+str(action_nr)+" State = "+str(State)+ " player = "+str(player))
						action_nr = np.random.randint(len(actions))
					time.sleep(2)
					

				print("action_nr = " + str(action_nr)+" actions = "+str(actions))
				NextState, Reward =  game_object.next_state_and_reward(player,State, actions[action_nr])
				self.draw_empty(actions[action_nr][1], actions[action_nr][2])
				self.draw_empty(actions[action_nr][3], actions[action_nr][4])
				#if promotion_piece != None:
				if len(actions[action_nr]) > 5:
					self.draw_piece(actions[action_nr][3], actions[action_nr][4], actions[action_nr][5])
				else:
					self.draw_piece(actions[action_nr][3], actions[action_nr][4], actions[action_nr][0])
				# castling - additional Rook move: 
				if (actions[action_nr][0] == self.game_object.King) & (actions[action_nr][2] == 4) &\
					(actions[action_nr][4] == 6):
					self.draw_empty(actions[action_nr][1], 7)
					self.draw_piece(actions[action_nr][1], 5, self.game_object.Rook)
				if (actions[action_nr][0] == self.game_object.King) & (actions[action_nr][2] == 4) &\
					(actions[action_nr][4] == 2):
					self.draw_empty(actions[action_nr][1], 0)
					self.draw_piece(actions[action_nr][1], 3, self.game_object.Rook)
				if (actions[action_nr][0] == self.game_object.King+self.game_object.BlackShift) & \
					(actions[action_nr][2] == 4) & (actions[action_nr][4] == 6):
					self.draw_empty(actions[action_nr][1], 7)
					self.draw_piece(actions[action_nr][1], 5, self.game_object.Rook+self.game_object.BlackShift)
				if (actions[action_nr][0] == self.game_object.King+self.game_object.BlackShift) &\
					(actions[action_nr][2] == 4) & (actions[action_nr][4] == 2):
					self.draw_empty(actions[action_nr][1], 0)
					self.draw_piece(actions[action_nr][1], 3, self.game_object.Rook+self.game_object.BlackShift)

				
				if game_object.end_of_game(Reward,step_number,State,action_nr):      # win or draw
					end_of_game = True
					if Reward == 1:
						winner = 1
					elif Reward == -1:
						winner = 2
					else:
						winner = 0
					self.draw_end_of_game(NextState,player,winner)
					
				self.draw_status(3-player,winner)

				if end_of_game:
					time.sleep(4)
					
				pg.display.update()
				fps = 30
				self.CLOCK.tick(fps)

				player = 3 - player
				State = NextState


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def play_with_strategy(game_object, strategy, str_player):
	game_name = type(game_object).__name__
	#class_name = "Interface_"+game_name
	#eval("interface_class = "+class_name+"()")
	if game_name == "Tictactoe":
		interface_class = Interface_Tictactoe(game_object)
	elif game_name == "Tictac_general":
		interface_class = Interface_Tictac_general(game_object)
	elif game_name == "Connect4":
		interface_class = Interface_Connect4(game_object)
	elif game_name == "Chess":
		interface_class = Interface_Chess(game_object)
	interface_class.play_with_strategy(game_object, strategy, str_player)
	