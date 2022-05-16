import numpy as np
import os
import config
from tkinter import *
from tkinter.messagebox import *
from typing import Optional
from game import Game
from mcts import MCTSPlayer


class GUI:
    def __init__(self, board_info):
        self.game: Game = board_info  # get the rule of the game for checking the validation of the human's actions.
        self.who_first: Optional[str] = None  # define which player to first put the black stone
        self.allow_human_click = False  # permission that human player (me) can click on the gui board to put a stone

    # Initially, draw cross
    def gui_draw_cross(self, x, y):
        cross_scale = 1.5  # cross scale '+'
        # define boundary coordinates
        screen_x, screen_y = self.standard_size * (x + 1), self.standard_size * (y + 1)
        # draw cell consisting of four cross
        self.gui_board.create_rectangle(screen_y - self.cross_size, screen_x - self.cross_size,
                                        screen_y + self.cross_size, screen_x + self.cross_size,
                                        fill=self.board_color, outline=self.board_color)
        hor_m, hor_n = [0, cross_scale] if y == 0 else [-cross_scale, 0] if y == config.GUI_BOARD_GRID - 1 else [
            -cross_scale, cross_scale]
        ver_m, ver_n = [0, cross_scale] if x == 0 else [-cross_scale, 0] if x == config.GUI_BOARD_GRID - 1 else [
            -cross_scale, cross_scale]
        # draw row
        self.gui_board.create_line(screen_y + hor_m * self.cross_size, screen_x, screen_y + hor_n * self.cross_size,
                                   screen_x)
        # draw col
        self.gui_board.create_line(screen_y, screen_x + ver_m * self.cross_size, screen_y,
                                   screen_x + ver_n * self.cross_size)

    # Draw stone(s)
    def gui_draw_stone(self, x, y, current_stone_color):
        screen_x, screen_y = self.standard_size * (x + 1), self.standard_size * (y + 1)
        screen_x += 0.5 * self.standard_size
        screen_y += 0.5 * self.standard_size
        self.gui_board.create_oval(screen_y - self.stone_size, screen_x - self.stone_size,
                                   screen_y + self.stone_size, screen_x + self.stone_size,
                                   fill=current_stone_color)
        # self.gui_board.create_text(screen_y, screen_x, text=(len(self.board_info.occupied_moves) - 4) + 1, fill='red')

    # Draw board, including cross and, horizontal and vertical lines.
    def gui_draw_board(self):
        # 根据棋盘尺寸大小循环
        [self.gui_draw_cross(x, y) for y in range(config.GUI_BOARD_GRID) for x in
         range(config.GUI_BOARD_GRID)]
        # By default, there are four cells occupied initially.
        self.gui_draw_stone(3, 3, 'white')
        self.gui_draw_stone(3, 4, 'black')
        self.gui_draw_stone(4, 4, 'white')
        self.gui_draw_stone(4, 3, 'black')
        # self.gui_draw_stone(0, 0, 'white')

    """
    Human vs Human mode
    Once clicked, black stone is the first stone to be put.
    """
    def gui_opt_human_start_btn(self):
        self.human_vs_human_mode = True
        self.who_first = 'player1'  # Player 1 go first, which represents black correspond to current_player_id = 1.
        self.allow_human_click = True
        self.gui_draw_board()  # 重置界面
        self.turn_tips.config(text="Black Round")
        self.game.initialize_board_info(self.who_first)  # 即result_id或id=1为黑棋 反之=2为白棋

    """
    Human vs AI mode
    This mode is available only if a model is trained.
    We can choose black/white side against AI trained; therefore, two buttons are click-allowed.
    """
    # Pick black side
    def gui_opt_black_btn(self):
        self.human_vs_human_mode = False
        self.who_first = 'player1'  # Let the player 1 goes first
        self.allow_human_click = True
        self.gui_draw_board()  # Initialize board and visually
        self.turn_tips.config(text="Black Round")
        self.game.initialize_board_info(who_first=self.who_first)

    # Pick white side
    def gui_opt_white_btn(self):
        self.human_vs_human_mode = False
        self.who_first = 'player2'  # Let the player 2 goes first
        self.allow_human_click = False
        self.gui_draw_board()  # Initialize board visually
        self.turn_tips.config(text="Black Round")
        self.gui_board.update()
        self.game.initialize_board_info(who_first=self.who_first)  # Initialize game (not GUI)
        current_player_id = self.game.get_current_player_id()  # Get the player id who goes the next round
        if current_player_id == 2:  # We always set the AI's id to 2
            AI = self.mcts_player
            move_id = AI.choose_move(self.game)
            gui_loc_y, gui_loc_x = self.game.move_2_location(move_id)
            flips = self.gui_draw_flips('black', 'White Round', gui_loc_y, gui_loc_x)
            self.game.move(move_id, flips)
            self.gui_board.update()
            self.turn_tips.config(text="White Round")
            self.allow_human_click = True

    # Draw the game result (Win, lose, draw ...)
    def gui_draw_center_result_text(self, text):
        width, height = int(self.gui_board['width']), int(self.gui_board['height'])
        self.gui_board.create_text(int(width / 2), int(height / 2), text=text, font=("黑体", 30, "bold"), fill="red")

    """
    For othello, flipping the stones from black/white to white/black is common.
    """
    def gui_draw_flips(self, current_color, next_move_tip, gui_loc_y, gui_loc_x):
        flips = self.game.is_valid_move(current_color, gui_loc_y, gui_loc_x)
        if flips:
            self.gui_draw_stone(gui_loc_y, gui_loc_x, current_color)
            for x, y in flips:
                self.gui_draw_stone(x, y, current_color)
            self.turn_tips.config(text=next_move_tip)
        return flips

    # Event that clicks on the board
    def gui_click_board(self, event):
        # Convert the position clicked to the position of board
        gui_loc_y, gui_loc_x = int((event.y - self.cross_size - 0.5 * self.standard_size) / self.standard_size), int(
            (event.x - self.cross_size - 0.5 * self.standard_size) / self.standard_size)
        if gui_loc_y >= 8 or gui_loc_x >= 8:  # Not allowed to click outside the boundary
            return
        if not self.allow_human_click:  # Check if the human click is valid
            return
        human_move = gui_loc_y * 8 + gui_loc_x  # Get the position of board according to event click
        if human_move not in self.game.non_occupied_stones:  # Not allowed to click the cell occupied
            return
        current_player_id = self.game.get_current_player_id()  # Get the player id who goes the current round

        # Human vs human: Black round
        if current_player_id == 1 and self.human_vs_human_mode is True:
            current_color = 'black' if self.who_first == 'player1' else 'white'
            next_move_tip = 'White Round' if self.who_first == 'player1' else 'Black Round'
            flips = self.gui_draw_flips(current_color, next_move_tip, gui_loc_y, gui_loc_x)
            if flips:
                will_pass_flag = self.game.move(human_move, flips)
                if will_pass_flag:
                    self.turn_tips.config(text='White Round')
                    showinfo(title='Pass', message='White pass')
                    will_double_pass = self.game.move(-1)
                    self.turn_tips.config(text='Black Round')
                    if will_double_pass:
                        showinfo(title='Pass', message='Black pass')
                        self.game.move(-1)
            self.gui_board.update()
            # check the result of the game
            status = self.game.get_game_status()  # 1 2 3 -1
            if status in (1, 2, 3):
                self.gui_draw_game_result(status, mode='hh')
                return
        # Human vs human: White round
        if current_player_id == 2 and self.human_vs_human_mode is True:
            current_color = 'black' if self.who_first == 'player2' else 'white'
            next_move_tip = 'White Round' if self.who_first == 'player2' else 'Black Round'
            flips = self.gui_draw_flips(current_color, next_move_tip, gui_loc_y, gui_loc_x)
            if flips:
                will_pass_flag = self.game.move(human_move, flips)
                if will_pass_flag:
                    self.turn_tips.config(text='Black Round')
                    showinfo(title='Pass', message='Black pass')
                    will_double_pass = self.game.move(-1)
                    self.turn_tips.config(text='White Round')
                    if will_double_pass:
                        showinfo(title='Pass', message='White pass')
                        self.game.move(-1)
            self.gui_board.update()
            status = self.game.get_game_status()
            # check the result of the game
            if status in (1, 2, 3):
                self.gui_draw_game_result(status, mode='hh')
                return

        # Human vs AI (MCTS): Human turn
        pass_flag = True
        while pass_flag and self.human_vs_human_mode is False:
            if current_player_id == 1 and not self.human_vs_human_mode:
                current_color = 'black' if self.who_first == 'player1' else 'white'
                next_move_tip = 'White Round' if self.who_first == 'player1' else 'Black Round'
                if self.game.get_available_moves() == [-1]:
                    showinfo(title='Pass', message=f'{current_color} pass')
                    self.turn_tips.config(text=next_move_tip)
                    self.game.move(-1)
                else:
                    flips = self.gui_draw_flips(current_color, next_move_tip, gui_loc_y, gui_loc_x)
                    if not flips:
                        return
                    self.game.move(human_move, flips)
                self.allow_human_click = False if self.human_vs_human_mode is False else True
                self.gui_board.update()
                # check the result of the game
                status = self.game.get_game_status()  # 1 2 3 -1
                if status in (1, 2, 3):
                    self.gui_draw_game_result(status, mode='ha')
                    return

            # Human vs AI (MCTS): AI turn
            current_player_id = self.game.get_current_player_id()
            if current_player_id == 2 and not self.human_vs_human_mode:
                current_color = 'white' if self.who_first == 'player1' else 'black'
                next_move_tip = 'Black Round' if self.who_first == 'player1' else 'White Round'
                if self.game.get_available_moves() == [-1]:
                    showinfo(title='Pass', message=f'{current_color} pass')
                    self.turn_tips.config(text=next_move_tip)
                    will_pass_flag = self.game.move(-1)
                else:
                    AI = self.mcts_player
                    move_id = AI.choose_move(self.game)
                    gui_loc_y, gui_loc_x = self.game.move_2_location(move_id)
                    flips = self.gui_draw_flips(current_color, next_move_tip, gui_loc_y, gui_loc_x)
                    will_pass_flag = self.game.move(move_id, flips)
                pass_flag = True if will_pass_flag else False
                self.allow_human_click = True
                self.gui_board.update()
                status = self.game.get_game_status()
                if status in (1, 2, 3):
                    self.gui_draw_game_result(status, mode='ha')
                    return

    # Exit the GUI
    @staticmethod
    def gui_quit_game():
        os._exit(1)

    def gui_draw_game_result(self, status, mode='hh'):
        self.allow_human_click = False
        text = self.gui_get_text_from_game_status(status=status, mode=mode)
        self.gui_draw_center_result_text(text)
        self.turn_tips.config(text="Waiting...")

    """
    Get the result of the game according to the mode chosen
    """
    def gui_get_text_from_game_status(self, status, mode='hh'):
        text = None
        if mode == 'hh':
            if status == 1:
                text = "Black wins!"
            elif status == 2:
                text = "White wins!"
            elif status == 3:
                text = "Draw"
            else:
                text = "Error occured in {} status!".format(mode)
        elif mode == 'ha':
            # print(self.game.get_current_player_id())
            if self.game.get_current_player_id() == 1:
                text = "AI wins!" if status != 3 else "Draw"
            else:
                text = "You win!" if status != 3 else "Draw"
        else:
            text = 'Error occurred in mode!'
        return text

    """
    GUI
    Define some basic properties.
    Design the layout and structure of the board game GUI.
    """
    def gui(self, mcts_player):
        self.board_size = config.GUI_BOARD_GRID
        self.mcts_player: MCTSPlayer = mcts_player # AI (MCTS) player
        self.human_vs_human_mode = False  # True for human vs human, while false for human vs AI
        # GUI style
        sidebar_color = "Moccasin"  # Color of the side bar
        btn_font = ("Arial", 12, "bold")  # Button font style
        self.standard_size = 40  # GUI window size
        self.board_color = "Tan"  # GUI board color
        self.cross_size = self.standard_size / 2  # Cross size
        self.stone_size = self.standard_size / 3  # Stone size
        self.allow_human_click = False

        # Create Tkinter
        root = Tk()
        root.title("Othello")
        root.resizable(width=False, height=False)  # Not allowed to drag the window to change its size.
        # Layout design
        gui_sidebar = Frame(root, highlightthickness=0, bg=sidebar_color)
        gui_sidebar.pack(fill=BOTH, ipadx=10, side=RIGHT)  # ipadx 加宽度padding
        btn_opt_black = Button(gui_sidebar, text="Black side", command=self.gui_opt_black_btn, font=btn_font)
        btn_opt_white = Button(gui_sidebar, text="White side", command=self.gui_opt_white_btn, font=btn_font)
        btn_opt_human_play_start = Button(gui_sidebar, text="Start game", command=self.gui_opt_human_start_btn, font=btn_font)
        # self.btn_opt_human_play_save = Button(gui_sidebar, text="保存棋局", command=self.gui_opt_human_save_btn, font=btn_font, state=DISABLED)
        btn_opt_quit = Button(gui_sidebar, text="Exit", command=self.gui_quit_game, font=btn_font)
        self.turn_tips = Label(gui_sidebar, text="Waiting", bg=sidebar_color, font=("Arial", 18, "bold"), fg="red4")
        two_human_play_label = Label(gui_sidebar, text="Human vs human", bg=sidebar_color, font=("Arial", 12, "bold"))
        machine_man_play_label = Label(gui_sidebar, text="Human vs AI", bg=sidebar_color, font=("Arial", 12, "bold"))
        # Show the layout
        two_human_play_label.pack(side=TOP, padx=20, pady=5)
        btn_opt_human_play_start.pack(side=TOP, padx=20, pady=10)
        # self.btn_opt_human_play_save.pack(side=TOP, padx=20, pady=10)
        machine_man_play_label.pack(side=TOP, padx=20, pady=10)
        btn_opt_black.pack(side=TOP, padx=20, pady=5)
        btn_opt_white.pack(side=TOP, padx=20, pady=10)
        btn_opt_quit.pack(side=BOTTOM, padx=20, pady=10)
        self.turn_tips.pack(side=TOP, expand=YES, fill=BOTH, pady=10)
        self.gui_board = Canvas(root, bg=self.board_color, width=(config.GUI_BOARD_GRID + 1) * self.standard_size,
                                height=(config.GUI_BOARD_GRID + 1) * self.standard_size, highlightthickness=0)
        self.gui_draw_board()  # Init board
        self.gui_board.pack()
        self.gui_board.bind("<Button-1>", self.gui_click_board)  # Mouse event (left)
        root.mainloop()  # Event loop

    """
    Open the GUI for 1. human vs human mode; 2. human vs AI mode
    The Self-play, training and evaluation modules are run in training.py
    """
    def start_game(self, mcts_player: MCTSPlayer):
        self.gui(mcts_player)
