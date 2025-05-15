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

class Interface_Tictac_general(Interface_Tictactoe):
	def __init__(self,game_object):

		self.num_of_rows = game_object.num_of_rows
		self.num_of_columns = game_object.num_of_columns
		self.num_of_stones = game_object.num_of_stones
		self.if_adjacent = game_object.if_adjacent

		self.width = 50*self.num_of_columns

		# to set height of the game window
		self.height = 50*self.num_of_rows

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
	