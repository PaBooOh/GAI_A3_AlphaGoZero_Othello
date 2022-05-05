# -*- coding: utf-8 -*-
import numpy as np
from tkinter import *
from tkinter.messagebox import *
import os
import config
from typing import Optional, Union, Any, Set, List, Tuple, Dict


class Foundation(object):
    def __init__(self, board_size):
        self.board_size = board_size  # 棋盘尺寸
        self.aggregate_move_count = 0  # 总棋子数
        self.avail_move_list = [i for i in range(8 ** 2)]  # 一维,可落子位置信息,初始可落子序号
        self.all_move_list = [27, 28, 35, 36]  # 下过的棋子(序号)
        self.all_player_id_list = []  # 落子玩家序号记录
        self.current_player_is_black = None  # 当前将要落子的玩家是否是黑棋
        # Othello
        self.board = self.getNewBoard() # 8*8二维列表，存放white black信息
        self.next_state_avail_moves = None
        self.next_state_avail_moves_id = None
        self.black_id_list = None
        self.white_id_list = None
        self.double_pass = False

    def getNewBoard(self):
        board = []
        for i in range(8):
            board.append(['none']*8)
        for x in range(8):
            for y in range(8):
                board[x][y] = 'none'
        # 初始四颗棋子
        board[3][3] = 'white'
        board[3][4] = 'black'
        board[4][3] = 'black'
        board[4][4] = 'white'
        return board

    def isOnBoard(self, x, y):
        return x >= 0 and x <= 7 and y >= 0 and y <= 7

    def isValidMove(self, tile, xstart, ystart):
        "是否合法， 如果合法返回需要翻转的棋子列表"
        if not self.isOnBoard(xstart, ystart) or self.board[xstart][ystart] != 'none':
            return False
        # 临时将tile放到指定位置
        self.board[xstart][ystart] = tile
        if tile == 'black':
            otherTile = 'white'
        else:
            otherTile = 'black'
        tilesToFlip = []
        for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x, y = xstart, ystart
            x += xdirection
            y += ydirection
            if self.isOnBoard(x, y) and self.board[x][y] == otherTile:
                x += xdirection
                y += ydirection
                if not self.isOnBoard(x, y):
                    continue
                # 一直走到出界或者不是对方棋子的位置
                while self.board[x][y] == otherTile:
                    x += xdirection
                    y += ydirection
                    if not self.isOnBoard(x, y):
                        break
                # 出界了 则没有棋子要反转 OXXXXXXX
                if not self.isOnBoard(x, y):
                    continue
                # 是自己的棋子 OXXXXXXO
                if self.board[x][y] == tile:
                    while True:
                        x -= xdirection
                        y -= ydirection
                        # 回到起点则结束
                        if x == xstart and y == ystart:
                            break
                        # 需要翻转的棋子
                        tilesToFlip.append([x, y])
        # 将前面临时放上去的棋子去掉，还原棋盘
        self.board[xstart][ystart] = 'none'
        # 没有要被翻转的棋子， 则该位置非法
        if len(tilesToFlip) == 0:
            return False
        return tilesToFlip

    def getValidMoves(self, tile):
        """获取可落子的位置"""
        validMoves = []
        for x in range(8):
            for y in range(8):
                if self.isValidMove(tile, x, y) != False:
                    validMoves.append([x, y])
        return validMoves

    def getScoreOfBoard(self):
        """获取棋盘上黑白双方的棋子数"""
        xscore = 0
        oscore = 0
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == 'black':
                    xscore += 1
                if self.board[x][y] == 'white':
                    oscore += 1
        return {'black': xscore, 'white': oscore}

    def isGameOver(self):
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == 'none':
                    return False
        return True

    # 重置棋盘信息
    def build_board(self, who_first='player1'):
        self.current_player_id = 1 if who_first == 'player1' else 2  # 设定第(1/2)个玩家为先手, 用于人机对战设置
        self.current_player_is_black = True
        self.avail_move_list = [i for i in range((self.board_size-1) ** 2)]  # 初始可落子序号
        self.all_move_list = [27, 28, 35, 36]  # 下过的棋子(序号)
        self.avail_move_list = [i for i in self.avail_move_list if i not in self.all_move_list] # 除去中间四个
        self.all_player_id_list = []  # 落子玩家序号记录
        # 初始当前棋盘落子数
        self.aggregate_move_count = 0
        ##########
        self.board = self.getNewBoard()  # 8*8二维列表，存放white black信息
        self.next_state_avail_moves = None
        self.next_state_avail_moves_id = None
        self.black_id_list = None
        self.white_id_list = None
        self.double_pass = False

    # 落子位置序号 -> 棋局矩阵坐标
    def move_2_location(self, move_id):
        y = move_id // (self.board_size - 1)
        x = move_id % (self.board_size - 1)
        return y, x

    # [[0,1],[1,1], ..., [], ...] --> [1, 9, ..., x, ...]
    def locationlist_2_moveList(self, location_list):
        move_list = []
        for location in location_list:
            move = location[0] * 8 + location[1]  # row * 8 + col
            move_list.append(move)
        return move_list
    
    # [1, 9, ..., x, ...] --> [[0,1],[1,1], ..., [], ...]
    def moveList_2_locationlist(self, move_list):
        location_list = []
        for move_id in move_list:
            y, x = self.move_2_location(move_id)
            location_list.append((y, x))
        return location_list

    # self.board 信息 --> 黑棋id列表和白棋id列表
    def board2tileid(self):
        black_id_list = []
        white_id_list = []
        for row in range(8):
            for col in range(8):
                tile_id = row * 8 + col
                if self.board[row][col] == 'black':
                    black_id_list.append(tile_id)
                if self.board[row][col] == 'white':
                    white_id_list.append(tile_id)
        return black_id_list, white_id_list

    # 获取当前棋局局面的特征平面
    def getFeaturePlane(self):
        # 返回当前棋盘状态的?个不同的特征平面（? depth）用于输入神经网络
        feature_plane = np.full((config.FEATURE_PLANE_NUM, self.board_size, self.board_size), 0.0)  # 定义4个平面(三维)
        if len(self.all_move_list) > 4:
            # 所有move+所有交替玩家的id 取得两个玩家各自的move_id(list都必须转为np)
            all_move = np.array(self.getAllMoveList())
            player1_move = all_move[np.array(self.getAllPlayerIdList()) != self.current_player_id]
            player2_move = all_move[np.array(self.getAllPlayerIdList()) == self.current_player_id]
            black_loc_list = self.moveList_2_locationlist(self.black_id_list)
            black_y_list = [black_loc[0] for black_loc in black_loc_list]
            black_x_list = [black_loc[1] for black_loc in black_loc_list]
            white_loc_list = self.moveList_2_locationlist(self.white_id_list)
            white_y_list = [white_loc[0] for white_loc in white_loc_list]
            white_x_list = [white_loc[1] for white_loc in white_loc_list]
            # 由落子顺序和玩家顺序 获取特征平面
            # Plane 1: Black stones are 1.0, others 0.0.
            feature_plane[0][black_y_list, black_x_list] = 1.0
            # Plane 2: White stones are 1.0, others 0.0.
            feature_plane[1][white_y_list, white_x_list] = 1.0
            # 特征平面3：当前将要落子的玩家的对手的历史位置（对手的上1步）
            feature_plane[2][self.board_size - self.getLastMoveId() // self.board_size - 1, self.getLastMoveId() % self.board_size] = 1.0
            # 特征平面4: 当前将要落子的是否是黑棋？是则棋盘全为1，否则为0；
            feature_plane[3] = 1.0 if len(self.all_move_list) % 2 == 0 else 0.0
            # 返回当前棋局状态所表征的特征平面
        return feature_plane

    # 更新棋盘数组 再返回给gui显示
    def step(self, move, tilesToFlip):
        self.aggregate_move_count += 1  # 记录总步数 用于gui
        # 下一步棋就从可落子数组移除当前下的那个
        self.avail_move_list.remove(move)
        self.all_move_list.append(move)
        self.all_player_id_list.append(self.current_player_id)
        # 获得落子序号的棋盘坐标
        loc_y, loc_x = self.move2location(move) # loc_y是行号 loc_x是列号

        # ****************** 黑白棋
        if self.current_player_id == 1:
            tile = 'black'
            othertile = 'white'
        if self.current_player_id == 2:
            tile = 'white'
            othertile = 'black'
        self.board[loc_y][loc_x] = tile
        for x,y in tilesToFlip:  # 需要翻转的棋子进行变色
            self.board[x][y] = tile

        print('Player_ids: ', self.all_player_id_list)
        # print(self.avail_move_list)
        # 落子后 获取黑棋特征/白棋特征到 属性中
        black_id_list, white_id_list = self.board2tileid()
        self.black_id_list = black_id_list
        self.white_id_list = white_id_list
        print('black_id_list',self.black_id_list)
        print('white_id_list', self.white_id_list)
        
        # ******************
        # 一方落子后 更新当前将要落子玩家序号
        if not self.isGameOver():
            next_state_avail_moves = self.getValidMoves(othertile)
            self.current_player_id = 1 if self.current_player_id == 2 else 2
            if next_state_avail_moves != []:
                self.next_state_avail_moves = next_state_avail_moves
                self.next_state_avail_moves_id = self.locationlist_2_moveList(self.next_state_avail_moves)
                print('next_state_avail_moves_id_1:',self.next_state_avail_moves_id)
                print()
            else:
                self.all_player_id_list.append(self.current_player_id)
                self.next_state_avail_moves = [-1]
                self.next_state_avail_moves_id = [-1]
                print('next_state_avail_moves_id_2:', self.next_state_avail_moves_id)
                print()
                if not self.isGameOver():
                    showinfo(title='Pass', message='Pass')
        else:
            self.next_state_avail_moves_id = []
            self.next_state_avail_moves = []
            print('next_state_avail_moves_id_5:', self.next_state_avail_moves_id)
            print()

    # 黑白棋判断输赢
    def getGameStatus(self):
        if not self.isGameOver:
            return -1
        who_win = None
        if self.isGameOver() or self.double_pass == True:
            scoreBlack = self.getScoreOfBoard()['black']
            scoreWhite = self.getScoreOfBoard()['white']
            if scoreBlack > scoreWhite:
                who_win = 1
            if scoreWhite > scoreBlack:
                who_win = 2
            if scoreBlack == scoreWhite:
                who_win = 3
        return who_win

    # 获取当前将要落子的玩家id
    def getCurrentPlayerId(self) -> int:
        return self.current_player_id

    def getAllMoveList(self):
        return self.all_move_list

    def getAllPlayerIdList(self):
        return self.all_player_id_list

    # 获得倒数第n步的落子序号
    def getLastMoveId(self, n=1):
        return self.all_move_list[-n] if n <= len(self.all_move_list) else -1

    # 获取当前将要落子的玩家是黑棋还是白棋
    def isCurrentPlayerBlack(self) -> bool:
        return self.current_player_is_black


class Game(object):
    def __init__(self, board_info):
        self.board_info: Foundation = board_info
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
        hor_m, hor_n = [0, cross_scale] if y == 0 else [-cross_scale, 0] if y == self.board_info.board_size - 1 else [
            -cross_scale, cross_scale]
        ver_m, ver_n = [0, cross_scale] if x == 0 else [-cross_scale, 0] if x == self.board_info.board_size - 1 else [
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
        screen_x += 0.5*self.standard_size
        screen_y += 0.5*self.standard_size
        # print(screen_x, screen_y)
        self.gui_board.create_oval(screen_y - self.stone_size, screen_x - self.stone_size,
                                   screen_y + self.stone_size, screen_x + self.stone_size,
                                   fill = current_stone_color)

    # 画棋盘
    def gui_draw_board(self):
        # 根据棋盘尺寸大小循环
        [self.gui_draw_cross(x, y) for y in range(self.board_info.board_size) for x in
         range(self.board_info.board_size)]
        self.gui_draw_stone(3,3,'white')
        self.gui_draw_stone(3,4,'black')
        self.gui_draw_stone(4,4,'white')
        self.gui_draw_stone(4,3,'black')

    # Human vs Human
    def gui_opt_human_start_btn(self):
        self.two_human_play_mode = True
        self.who_first = 'player1'  # 即设定玩家1为先手
        self.allow_human_click = True
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.board_info.build_board(self.who_first)  # 即result_id或id=1为黑棋 反之=2为白棋

    # 初始化界面信息（选择黑色）
    def gui_opt_black_btn(self):
        self.two_human_play_mode = False
        self.who_first = 'player1'  # 即设定玩家1为先手
        self.allow_human_click = True
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.board_info.build_board(who_first=self.who_first)

    # 初始化界面信息（选择白色）
    def gui_opt_white_btn(self):
        self.two_human_play_mode = False
        self.who_first = 'player2'  # 即设定玩家2为先手
        self.allow_human_click = False
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.gui_board.update()
        self.board_info.build_board(who_first=self.who_first)  # 人类选白棋， 初始化棋盘

        current_player_id = self.board_info.getCurrentPlayerId()  # 获取当前将要落子的玩家id
        # 人类选择白棋按钮后,让电脑先走一步后再进入鼠标左键的绑定事件
        if current_player_id == 2:  # 电脑的id都被设置为2
            alternate_player_obj = self.ai_obj
            mcts_move = alternate_player_obj.choose_move(self.board_info)
            gui_pos_row, gui_pos_col = self.board_info.move_2_location(mcts_move)
            self.gui_draw_stone(gui_pos_row, gui_pos_col, 'black')
            self.board_info.step(mcts_move)
            self.gui_board.update()
            self.label_tips.config(text="白棋回合")
            self.allow_human_click = True

    # 棋局结束在canvas中间显示胜负提醒
    def gui_draw_center_text(self, text):
        width, height = int(self.gui_board['width']), int(self.gui_board['height'])
        self.gui_board.create_text(int(width / 2), int(height / 2), text=text, font=("黑体", 30, "bold"), fill="red")

    # 棋盘Canvas的click事件
    def gui_click_board(self, event):
        # 获取鼠标点击的canvas坐标并用int强制转化为网格坐标
        gui_pos_row, gui_pos_col = int((event.y - self.cross_size - 0.5*self.standard_size) / self.standard_size), int(
            (event.x - self.cross_size - 0.5*self.standard_size) / self.standard_size)
        # print('**** new click ****')
        # print('click at: ', gui_pos_row,gui_pos_col)  # (0,0) ~ (7,7)
        # 防止超出边界
        if gui_pos_row >= 8 or gui_pos_col >= 8: # 黑白棋修改
            return
        # 防止其他禁止情况却能落子
        if not self.allow_human_click:
            return
        # 鼠标的点击坐标转化为 具体落子数字的格式
        self.human_move = gui_pos_row * 8 + gui_pos_col
        # 已经下过的地方不能再下
        if self.human_move not in self.board_info.avail_move_list:
            return
        current_player_id = self.board_info.getCurrentPlayerId()  # 当前将要落子的玩家id
        # 人机对弈:玩家走子
        if current_player_id == 1 and self.two_human_play_mode is False:
            if self.human_move in self.board_info.avail_move_list:
                # 人类选择的先后手判断
                current_color = 'black' if self.who_first == 'player1' else 'white'
                next_move_tip = '白棋回合' if self.who_first == 'player1' else '黑棋回合'
                self.gui_draw_stone(gui_pos_row, gui_pos_col, current_color)
                self.label_tips.config(text=next_move_tip)
                self.allow_human_click = False if self.two_human_play_mode is False else True
                self.board_info.step(self.human_move)
                self.gui_board.update()
                winner_id = self.board_info.gobang_rules()  # 1 2 3 -1
                if winner_id in (1, 2, 3):
                    self.allow_human_click = False
                    text = "人类获胜" if winner_id != 3 else "平局"
                    self.gui_draw_center_text(text)
                    self.label_tips.config(text="等待中")
                    return
        # 双人对战（黑棋走子）
        if current_player_id == 1 and self.two_human_play_mode is True:
            if self.human_move in self.board_info.avail_move_list: # 判断是不是是没下过的地方
                # 人类选择的先后手判断
                current_color = 'black' if self.who_first == 'player1' else 'white'
                next_move_tip = '白棋回合' if self.who_first == 'player1' else '黑棋回合'
                if self.board_info.isValidMove(current_color,gui_pos_row,gui_pos_col): # 是否符合黑白棋落子规则
                        tilesToFlip = self.board_info.isValidMove(current_color,gui_pos_row,gui_pos_col)
                        self.gui_draw_stone(gui_pos_row, gui_pos_col, current_color)
                        for x,y in tilesToFlip:
                            self.gui_draw_stone(x,y,current_color)
                        self.label_tips.config(text=next_move_tip)
                # 后台进行下棋步骤
                try:
                    self.board_info.step(self.human_move,tilesToFlip)
                except:
                    pass
                # 下完当前步后 如果下一步没有合法位置 则本方继续
                if self.board_info.next_state_avail_moves == [-1]:
                    self.label_tips.config(text='黑棋回合')
                    next_state_avail_moves = self.board_info.getValidMoves('black')
                    self.board_info.next_state_avail_moves = next_state_avail_moves
                    self.board_info.next_state_avail_moves_id = self.board_info.locationlist_2_moveList(self.board_info.next_state_avail_moves)
                    self.board_info.current_player_id = 1
                    if self.board_info.next_state_avail_moves == []:
                        self.board_info.double_pass = True
                    print('black_id_list', self.board_info.black_id_list)
                    print('white_id_list', self.board_info.white_id_list)
                    print('next_state_avail_moves_id_pass1:', self.board_info.next_state_avail_moves_id)
                    print()
                self.gui_board.update()

                # 判断输赢
                winner_id = self.board_info.getGameStatus()  # 1 2 3
                if winner_id in (1, 2, 3):
                    self.allow_human_click = False
                    if winner_id == 1:
                        text = "黑棋获胜"
                    elif winner_id == 2:
                        text = "白棋获胜"
                    elif winner_id == 3:
                        text = "平局"
                    print('next_state_avail_moves_id_win1', self.board_info.next_state_avail_moves_id)
                    # print(self.board_info.get_all_player_id_list())
                    # print(len(self.board_info.get_all_player_id_list()))
                    self.gui_draw_center_text(text)
                    self.label_tips.config(text="等待中")
                    return

        # 人机对弈:AI走子
        current_player_id = self.board_info.getCurrentPlayerId()
        if current_player_id == 2 and self.two_human_play_mode is False:
            alternate_player_obj = self.ai_obj
            mcts_move = alternate_player_obj.choose_move(self.board_info)
            mcts_location = self.board_info.move_2_location(mcts_move)
            gui_pos_row, gui_pos_col = mcts_location[0], mcts_location[1]
            # 你选择先后手按钮 电脑相应需要改变
            current_color = 'white' if self.who_first == 'player1' else 'black'
            next_move_tip = '黑棋回合' if self.who_first == 'player1' else '白棋回合'
            self.gui_draw_stone(gui_pos_row, gui_pos_col, current_color)
            self.label_tips.config(text=next_move_tip)
            self.board_info.step(mcts_move)
            self.gui_board.update()
            self.allow_human_click = True
            winner_id = self.board_info.gobang_rules()
            if winner_id in (1, 2, 3):
                self.allow_human_click = False
                text = "电脑获胜" if winner_id != 3 else "平局"
                self.gui_draw_center_text(text)
                self.label_tips.config(text="等待中")
                return
        # 双人对战:白棋走子
        if current_player_id == 2 and self.two_human_play_mode is True:
            if self.human_move in self.board_info.avail_move_list:
                # 人类选择的先后手判断
                current_color = 'black' if self.who_first == 'player2' else 'white'
                next_move_tip = '白棋回合' if self.who_first == 'player2' else '黑棋回合'
                if self.board_info.isValidMove(current_color, gui_pos_row, gui_pos_col):
                    tilesToFlip = self.board_info.isValidMove(current_color, gui_pos_row, gui_pos_col)
                    self.gui_draw_stone(gui_pos_row, gui_pos_col, current_color)
                    for x,y in tilesToFlip:
                        self.gui_draw_stone(x, y, current_color)
                    self.label_tips.config(text = next_move_tip)

                try:
                    self.board_info.step(self.human_move,tilesToFlip)
                except:
                    # print('illegal move!!! ')
                    pass
                if self.board_info.next_state_avail_moves == [-1]:
                    self.label_tips.config(text='白棋回合')
                    next_state_avail_moves = self.board_info.getValidMoves('white')
                    self.board_info.next_state_avail_moves = next_state_avail_moves
                    self.board_info.next_state_avail_moves_id = self.board_info.locationlist_2_moveList(self.board_info.next_state_avail_moves)
                    self.board_info.current_player_id = 2
                    if self.board_info.next_state_avail_moves == []:
                        self.board_info.double_pass = True
                    print('black_id_list', self.board_info.black_id_list)
                    print('white_id_list', self.board_info.white_id_list)
                    print('next_state_avail_moves_id_pass2:', self.board_info.next_state_avail_moves_id)
                    print()
                self.gui_board.update()
                winner_id = self.board_info.getGameStatus()
                if winner_id in (1, 2, 3):
                    self.allow_human_click = False
                    if winner_id == 1:
                        text = "黑棋获胜"
                    elif winner_id == 2:
                        text = "白棋获胜"
                    elif winner_id == 3:
                        text = "平局"
                    print('next_state_avail_moves_id_win2', self.board_info.next_state_avail_moves_id)
                    self.gui_draw_center_text(text)
                    self.label_tips.config(text="等待中")
                    return

    # gui退出
    @staticmethod
    def gui_quit_game():
        os._exit(1)

    # 界面
    def gui(self, board_info: Foundation, mcts_player):
        self.board_size = board_info.board_size
        # 设置先手是玩家1还是玩家2（player?_obj)
        self.ai_obj = mcts_player
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
        self.gui_board = Canvas(root, bg=self.board_color, width=(self.board_size + 1) * self.standard_size,
                                height=(self.board_size + 1) * self.standard_size, highlightthickness=0)
        self.gui_draw_board()  # 初始化棋盘
        self.gui_board.pack()
        self.gui_board.bind("<Button-1>", self.gui_click_board)  # 绑定左键事件
        root.mainloop()  # 事件循环

    # 定义人机对战/电脑自我对弈训练

    # 打开gui界面进行人机对战和双人对战
    def start_game(self, mcts_player):
        self.gui(self.board_info, mcts_player)

    # 模型评估对战
    def model_play(self, latest_obj1, good_obj2, round_num):
        # 轮流设置先后手
        who_black, who_white = ['player1', 'player2'] if round_num % 2 == 0 else ['player2', 'player1']  # 交替先后手
        # 初始化棋盘信息
        self.board_info.build_board(who_black)
        # 两个模型对弈
        while True:
            mcts_player = latest_obj1 if who_black == 'player1' else good_obj2  # 玩家交替
            move_id = mcts_player.choose_move(self.board_info)  # mcts搜索
            self.board_info.step(move_id)  # 更新棋盘信息
            # 每落子一次就check一次
            result_id = self.board_info.gobang_rules()
            # 1=latest 2=good 3=tie
            if result_id in (1, 2, 3):
                return result_id

    # 自我对弈：生成一局棋谱-->状态集S、走子概率集Pi、价值集Z-->数据增强
    def self_play(self, mcts_player):
        # 每开一局都要初始化棋盘信息
        self.board_info.build_board()
        # S、Pi、Player_id
        states_list = []  # 一局游戏里的所有棋局状态集
        mcts_pi_list = []  # 一局游戏里的所有局面对应的每个落子位置的概率分布pi(由温度参数给出)
        # 电脑自我对弈
        while True:
            # 当前棋局局面送入mcts+nn 而后输出具体落子位置和落子概率
            move_id, pi = mcts_player.choose_move(self.board_info)
            # 下一步棋就存起来 (pi.reshape 与state相反)
            states_list.append(self.board_info.getFeaturePlane())  # 每个ndarr类型的局面的4个特征平面(3d)存入?_list
            mcts_pi_list.append(pi)  # 通过多次搜索（playout）后由softmax+tau得出扩展的子节点及其选择概率
            # 更新棋盘数组 记录落子
            self.board_info.step(move_id)
            # 每下一步check一次胜负
            result_id = self.board_info.getGameStatus()
            # 分出胜负： 1=player1 2=player2 3=tie
            if result_id in (1, 2, 3):
                # 下完一局后，记录每个状态下的z值
                z_list = np.zeros(self.board_info.aggregate_move_count)  # 一局一共走了多少步, 创建步数记录列表z_list
                # 不是平局:赋对应的胜负reward; 平局:赋全0
                if result_id != 3:
                    # 强化学习的奖惩原理, 主要考虑到无禁手的不平衡性, 可以自定义reward
                    if config.REWARD_CUSTOM_OPTIONS:
                        # 若是黑棋胜利
                        if self.board_info.isCurrentPlayerBlack is False:
                            # player_id必须用np.array()处理,否则只会赋-1.0,而1.0不会被赋
                            z_list[np.array(self.board_info.getAllPlayerIdList()) == result_id] = config.BLACK_WIN_SCORE  # 黑棋赢了
                            z_list[np.array(self.board_info.getAllPlayerIdList()) != result_id] = config.WHITE_LOSE_SCORE  # 白棋输了
                        # 若是白棋胜利
                        elif self.board_info.isCurrentPlayerBlack is True:
                            z_list[np.array(self.board_info.getAllPlayerIdList()) == result_id] = config.WHITE_WIN_SCORE  # 白棋赢了
                            z_list[np.array(self.board_info.getAllPlayerIdList()) != result_id] = config.BlACK_LOSE_SCORE  # 黑棋输了要扣更多分
                    else:  # 依据Zero围棋的reward
                        z_list[np.array(self.board_info.getAllPlayerIdList()) == result_id] = config.NORMAL_SCORE
                        z_list[np.array(self.board_info.getAllPlayerIdList()) != result_id] = -config.NORMAL_SCORE
                # 打包为(S、Pi、Z)一一对应的列表
                one_game_data = list(zip(states_list, mcts_pi_list, z_list))
                return one_game_data

    # 数据增强
    def expand_data(self, one_game_data, board_size):
        # play_data: [(state, pi, z), ..., ...]
        data_extension = []
        # mcts_pi -> 长度为size**2的向量
        # s/pi上下相反
        # 一局游戏包含的若干局面的3个数据 np.arr / np.arr / np.float64
        for one_state, mcts_pi, value_z in one_game_data:  # 遍历一局游戏的所有局面的数据(s,pi,v)
            # （1）正常翻转4次
            for i in range(1, 5):
                origin_rot_state_planes = np.array([np.rot90(one_plane, k=i) for one_plane in one_state])
                origin_rot_pi = np.rot90(np.flipud(np.reshape(mcts_pi, (board_size, board_size))), k=i)
                origin_one_state_data = (origin_rot_state_planes, np.flipud(origin_rot_pi).flatten(), value_z)  # pi多转了一次。。
                data_extension.append(origin_one_state_data)
            # （2）镜像
            mirror_state_planes = np.array([np.fliplr(one_plane) for one_plane in one_state])
            mirror_pi = np.fliplr(np.flipud(np.reshape(mcts_pi, (board_size, board_size))))  # 先取反与state一致
            # 镜像翻转4次
            for i in range(1, 5):
                mirror_rot_state_planes = np.array([np.rot90(one_plane, k=i) for one_plane in mirror_state_planes])
                mirror_rot_pi = np.rot90(mirror_pi, k=i)
                mirror_one_state_data = (mirror_rot_state_planes, np.flipud(mirror_rot_pi).flatten(), value_z)  # pi多转了一次。。
                data_extension.append(mirror_one_state_data)
        return data_extension