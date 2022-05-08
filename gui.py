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
        self.game: Game = board_info
        self.who_first: Optional[str] = None
        self.allow_human_click = False

    # 画棋盘和网格
    def gui_draw_cross(self, x, y):
        cross_scale = 1.5  # 交叉轴"+"的长度
        # 边界坐标
        screen_x, screen_y = self.standard_size * (x + 1), self.standard_size * (y + 1)
        # 画棋盘(a,b,c,d) -> (a,b)左上角坐标 (c,d)右下角坐标
        self.gui_board.create_rectangle(screen_y - self.cross_size, screen_x - self.cross_size,
                                        screen_y + self.cross_size, screen_x + self.cross_size,
                                        fill=self.board_color, outline=self.board_color)
        # 生成交叉点
        # 棋盘边缘的交叉点的x/y需设为0 其他情况就直接返回具体xyZ坐标
        hor_m, hor_n = [0, cross_scale] if y == 0 else [-cross_scale, 0] if y == config.GUI_BOARD_GRID - 1 else [
            -cross_scale, cross_scale]
        ver_m, ver_n = [0, cross_scale] if x == 0 else [-cross_scale, 0] if x == config.GUI_BOARD_GRID - 1 else [
            -cross_scale, cross_scale]
        # 画横线
        self.gui_board.create_line(screen_y + hor_m * self.cross_size, screen_x, screen_y + hor_n * self.cross_size,
                                   screen_x)
        # 画竖线
        self.gui_board.create_line(screen_y, screen_x + ver_m * self.cross_size, screen_y,
                                   screen_x + ver_n * self.cross_size)

    # 画棋子
    def gui_draw_stone(self, x, y, current_stone_color):
        screen_x, screen_y = self.standard_size * (x + 1), self.standard_size * (y + 1)
        screen_x += 0.5 * self.standard_size
        screen_y += 0.5 * self.standard_size
        self.gui_board.create_oval(screen_y - self.stone_size, screen_x - self.stone_size,
                                   screen_y + self.stone_size, screen_x + self.stone_size,
                                   fill=current_stone_color)
        # self.gui_board.create_text(screen_y, screen_x, text=(len(self.board_info.occupied_moves) - 4) + 1, fill='red')

    # 画棋盘
    def gui_draw_board(self):
        # 根据棋盘尺寸大小循环
        [self.gui_draw_cross(x, y) for y in range(config.GUI_BOARD_GRID) for x in
         range(config.GUI_BOARD_GRID)]
        self.gui_draw_stone(3, 3, 'white')
        self.gui_draw_stone(3, 4, 'black')
        self.gui_draw_stone(4, 4, 'white')
        self.gui_draw_stone(4, 3, 'black')
        # self.gui_draw_stone(0, 0, 'white')

    # Human vs Human
    def gui_opt_human_start_btn(self):
        self.two_human_play_mode = True
        self.who_first = 'player1'  # Player 1 go first, which represents black correspond to current_player_id = 1.
        self.allow_human_click = True
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.game.initialize_board_info(self.who_first)  # 即result_id或id=1为黑棋 反之=2为白棋

    # Human vs AI
    # 初始化界面信息（选择黑色）
    def gui_opt_black_btn(self):
        self.two_human_play_mode = False
        self.who_first = 'player1'  # 即设定玩家1为先手
        self.allow_human_click = True
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.game.initialize_board_info(who_first=self.who_first)

    # 初始化界面信息（选择白色）
    def gui_opt_white_btn(self):
        self.two_human_play_mode = False
        self.who_first = 'player2'  # 即设定玩家2为先手
        self.allow_human_click = False
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.gui_board.update()
        self.game.initialize_board_info(who_first=self.who_first)  # 人类选白棋， 初始化棋盘
        current_player_id = self.game.get_current_player_id()  # 获取当前将要落子的玩家id
        # 人类选择白棋按钮后,让电脑先走一步后再进入鼠标左键的绑定事件
        if current_player_id == 2:  # 电脑的id都被设置为2
            AI = self.mcts_player
            move_id = AI.choose_move(self.game)
            gui_loc_y, gui_loc_x = self.game.move_2_location(move_id)
            flips = self.gui_draw_flips('black', '白棋回合', gui_loc_y, gui_loc_x)
            self.game.move(move_id, flips)
            self.gui_board.update()
            self.label_tips.config(text="白棋回合")
            self.allow_human_click = True

    # 棋局结束在canvas中间显示胜负提醒
    def gui_draw_center_result_text(self, text):
        width, height = int(self.gui_board['width']), int(self.gui_board['height'])
        self.gui_board.create_text(int(width / 2), int(height / 2), text=text, font=("黑体", 30, "bold"), fill="red")

    def gui_draw_flips(self, current_color, next_move_tip, gui_loc_y, gui_loc_x):
        flips = self.game.is_valid_move(current_color, gui_loc_y, gui_loc_x)
        if flips:
            self.gui_draw_stone(gui_loc_y, gui_loc_x, current_color)
            for x, y in flips:
                self.gui_draw_stone(x, y, current_color)
            self.label_tips.config(text=next_move_tip)
        return flips

    # 棋盘Canvas的click事件
    def gui_click_board(self, event):
        # 获取鼠标点击的canvas坐标并用int强制转化为网格坐标
        gui_loc_y, gui_loc_x = int((event.y - self.cross_size - 0.5 * self.standard_size) / self.standard_size), int(
            (event.x - self.cross_size - 0.5 * self.standard_size) / self.standard_size)
        if gui_loc_y >= 8 or gui_loc_x >= 8:  # 黑白棋 防止超出边界
            return
        # 防止其他禁止情况却能落子
        if not self.allow_human_click:
            return
        # 鼠标的点击坐标转化为 具体落子数字的格式
        human_move = gui_loc_y * 8 + gui_loc_x
        # 已经下过的地方不能再下
        if human_move not in self.game.non_occupied_stones:
            return
        current_player_id = self.game.get_current_player_id()  # 当前将要落子的玩家id

        # 双人对战:黑棋走子
        if current_player_id == 1 and self.two_human_play_mode is True:
            # 人类选择的先后手判断
            current_color = 'black' if self.who_first == 'player1' else 'white'
            next_move_tip = '白棋回合' if self.who_first == 'player1' else '黑棋回合'
            flips = self.gui_draw_flips(current_color, next_move_tip, gui_loc_y, gui_loc_x)
            if flips:
                will_pass_flag = self.game.move(human_move, flips)
                if will_pass_flag:
                    self.label_tips.config(text='白棋回合')
                    showinfo(title='Pass', message='White pass')
                    will_double_pass = self.game.move(-1)
                    self.label_tips.config(text='黑棋回合')
                    if will_double_pass:
                        showinfo(title='Pass', message='Black pass')
                        self.game.move(-1)
            self.gui_board.update()
            # 判断输赢
            status = self.game.get_game_status()  # 1 2 3 -1
            if status in (1, 2, 3):
                self.gui_draw_game_result(status, mode='hh')
                return
        # 双人对战:白棋走子
        if current_player_id == 2 and self.two_human_play_mode is True:
            # 人类选择的先后手判断
            current_color = 'black' if self.who_first == 'player2' else 'white'
            next_move_tip = '白棋回合' if self.who_first == 'player2' else '黑棋回合'
            flips = self.gui_draw_flips(current_color, next_move_tip, gui_loc_y, gui_loc_x)
            if flips:
                will_pass_flag = self.game.move(human_move, flips)
                if will_pass_flag:
                    self.label_tips.config(text='黑棋回合')
                    showinfo(title='Pass', message='Black pass')
                    will_double_pass = self.game.move(-1)
                    self.label_tips.config(text='白棋回合')
                    if will_double_pass:
                        showinfo(title='Pass', message='White pass')
                        self.game.move(-1)
            self.gui_board.update()
            status = self.game.get_game_status()
            if status in (1, 2, 3):
                self.gui_draw_game_result(status, mode='hh')
                return

        # 人机对弈: 玩家走子
        if current_player_id == 1 and self.two_human_play_mode is False:
            # 人类选择的先后手判断
            current_color = 'black' if self.who_first == 'player1' else 'white'
            next_move_tip = '白棋回合' if self.who_first == 'player1' else '黑棋回合'
            flips = self.gui_draw_flips(current_color, next_move_tip, gui_loc_y, gui_loc_x)
            if not flips:
                return
            will_pass_flag = self.game.move(human_move, flips)
            if will_pass_flag:
                showinfo(title='Pass', message='Black pass')
            # self.gui_draw_stone(gui_loc_y, gui_loc_x, current_color)
            self.label_tips.config(text=next_move_tip)
            self.allow_human_click = False if self.two_human_play_mode is False else True
            self.gui_board.update()
            status = self.game.get_game_status()  # 1 2 3 -1
            if status in (1, 2, 3):
                self.gui_draw_game_result(status, mode='ha')
                return

        # 人机对弈: AI走子
        current_player_id = self.game.get_current_player_id()
        if current_player_id == 2 and self.two_human_play_mode is False:
            AI = self.mcts_player
            move_id = AI.choose_move(self.game)
            move_loc = self.game.move_2_location(move_id)
            gui_loc_y, gui_loc_x = move_loc[0], move_loc[1]
            # 你选择先后手按钮 电脑相应需要改变
            current_color = 'white' if self.who_first == 'player1' else 'black'
            next_move_tip = '黑棋回合' if self.who_first == 'player1' else '白棋回合'
            flips = self.gui_draw_flips(current_color, next_move_tip, gui_loc_y, gui_loc_x)
            # self.gui_draw_stone(gui_loc_y, gui_loc_x, current_color)
            self.label_tips.config(text=next_move_tip)
            self.game.move(move_id)
            self.gui_board.update()
            self.allow_human_click = True
            status = self.game.get_game_status()
            if status in (1, 2, 3):
                self.gui_draw_game_result(status, mode='ha')
                return

    # gui退出
    @staticmethod
    def gui_quit_game():
        os._exit(1)

    def gui_draw_game_result(self, status, mode='hh'):
        self.allow_human_click = False
        text = self.gui_get_text_from_game_status(status=status, mode=mode)
        self.gui_draw_center_result_text(text)
        self.label_tips.config(text="等待中")

    def gui_get_text_from_game_status(self, status, mode='hh'):
        text = None
        if mode == 'hh':
            if status == 1:
                text = "黑棋获胜"
            elif status == 2:
                text = "白棋获胜"
            elif status == 3:
                text = "平局"
            else:
                text = "Error occured in {} status!".format(mode)
        elif mode == 'ha':
            text = "电脑获胜" if status != 3 else "平局"
        else:
            text = 'Error occured in mode!'
        return text

    # 界面
    def gui(self, mcts_player):
        self.board_size = config.GUI_BOARD_GRID
        # 设置先手是玩家1还是玩家2（player?_obj)
        self.mcts_player: MCTSPlayer = mcts_player
        self.two_human_play_mode = False  # 确定是人机对弈还是双人对战
        # gui参数定义
        sidebar_color = "Moccasin"  # 侧边栏颜色
        btn_font = ("黑体", 12, "bold")  # 按钮文字样式
        self.standard_size = 40  # 设置标准尺寸
        self.board_color = "Tan"  # 棋盘颜色
        self.cross_size = self.standard_size / 2  # 交叉轴大小
        self.stone_size = self.standard_size / 3  # 棋子大小
        self.allow_human_click = False  # 是否允许人类玩家点击棋盘

        # gui初始化（tkinter)
        root = Tk()
        root.title("黑白棋")
        root.resizable(width=False, height=False)  # 窗口大小不允许拉动
        # 布局-定义
        gui_sidebar = Frame(root, highlightthickness=0, bg=sidebar_color)
        gui_sidebar.pack(fill=BOTH, ipadx=10, side=RIGHT)  # ipadx 加宽度padding
        btn_opt_black = Button(gui_sidebar, text="选择黑色", command=self.gui_opt_black_btn, font=btn_font)
        btn_opt_white = Button(gui_sidebar, text="选择白色", command=self.gui_opt_white_btn, font=btn_font)
        btn_opt_human_play_start = Button(gui_sidebar, text="开始游戏", command=self.gui_opt_human_start_btn, font=btn_font)
        # self.btn_opt_human_play_save = Button(gui_sidebar, text="保存棋局", command=self.gui_opt_human_save_btn, font=btn_font, state=DISABLED)
        btn_opt_quit = Button(gui_sidebar, text="退出游戏", command=self.gui_quit_game, font=btn_font)
        self.label_tips = Label(gui_sidebar, text="等待中", bg=sidebar_color, font=("黑体", 18, "bold"), fg="red4")
        two_human_play_label = Label(gui_sidebar, text="双人对战", bg=sidebar_color, font=("楷体", 12, "bold"))
        machine_man_play_label = Label(gui_sidebar, text="人机对战", bg=sidebar_color, font=("楷体", 12, "bold"))
        # 布局-显示
        two_human_play_label.pack(side=TOP, padx=20, pady=5)
        btn_opt_human_play_start.pack(side=TOP, padx=20, pady=10)
        # self.btn_opt_human_play_save.pack(side=TOP, padx=20, pady=10)
        machine_man_play_label.pack(side=TOP, padx=20, pady=10)
        btn_opt_black.pack(side=TOP, padx=20, pady=5)
        btn_opt_white.pack(side=TOP, padx=20, pady=10)
        btn_opt_quit.pack(side=BOTTOM, padx=20, pady=10)
        self.label_tips.pack(side=TOP, expand=YES, fill=BOTH, pady=10)
        self.gui_board = Canvas(root, bg=self.board_color, width=(config.GUI_BOARD_GRID + 1) * self.standard_size,
                                height=(config.GUI_BOARD_GRID + 1) * self.standard_size, highlightthickness=0)
        self.gui_draw_board()  # 初始化棋盘
        self.gui_board.pack()
        self.gui_board.bind("<Button-1>", self.gui_click_board)  # 绑定左键事件
        root.mainloop()  # 事件循环

    # 定义人机对战/电脑自我对弈训练

    # 打开gui界面进行人机对战和双人对战
    def start_game(self, mcts_player: MCTSPlayer):
        self.gui(mcts_player)
