import numpy as np
import config


class Game:
    def __init__(self, board_size):
        self.board_size = board_size  # 棋盘尺寸
        self.non_occupied_stones = list(set([i for i in range(self.board_size ** 2)]) - {27, 28, 35, 36})
        self.occupied_stones = [27, 28, 35, 36]  # 下过的棋子(序号)
        self.all_player_id_list = []  # 落子玩家序号记录
        self.current_player_is_black = None  # 当前将要落子的玩家是否是黑棋
        # Othello
        self.board = self.get_new_board()  # 8*8二维列表，存放white black信息
        self.next_state_avail_moves_loc = self.get_valid_moves('black')  # [[2, 3], [3, 2], [4, 5], [5, 4]]
        self.next_state_avail_moves_id = self.locations_2_moves(self.next_state_avail_moves_loc)
        self.black_id_list = [28, 35]
        self.white_id_list = [27, 36]
        self.passed = [False, False]

    def initialize_board_info(self, who_first='player1'):
        self.current_player_id = 1 if who_first == 'player1' else 2  # 设定第(1/2)个玩家为先手, 用于人机对战设置
        self.non_occupied_stones = list(set([i for i in range(self.board_size ** 2)]) - {27, 28, 35, 36})
        self.occupied_stones = [27, 28, 35, 36]  # 下过的棋子(序号)
        self.non_occupied_stones = [i for i in self.non_occupied_stones if i not in self.occupied_stones]  # 除去中间四个
        self.all_player_id_list = []  # 落子玩家序号记录
        # 初始当前棋盘落子数
        self.board = self.get_new_board()  # 8*8二维列表，存放white black信息
        self.next_state_avail_moves_loc = self.get_valid_moves('black')  # [[2, 3], [3, 2], [4, 5], [5, 4]]
        self.next_state_avail_moves_id = self.locations_2_moves(self.next_state_avail_moves_loc)
        self.black_id_list = [28, 35]
        self.white_id_list = [27, 36]
        self.passed = [False, False]
        self.current_player_is_black = True
        # self.print_game_information()

    def get_new_board(self):
        board = []
        for _ in range(self.board_size):
            board.append(['none'] * 8)
        for x in range(self.board_size):
            for y in range(self.board_size):
                board[x][y] = 'none'
        # 初始四颗棋子
        board[3][3] = 'white'
        board[3][4] = 'black'
        board[4][3] = 'black'
        board[4][4] = 'white'
        return board

    def is_valid_move(self, tile: str, loc_y, loc_x):
        "是否合法， 如果合法返回需要翻转的棋子列表"
        if not self.is_on_board(loc_y, loc_x) or self.board[loc_y][loc_x] != 'none':
            return False
        # 临时将tile放到指定位置
        self.board[loc_y][loc_x] = tile
        if tile == 'black':
            otherTile = 'white'
        else:
            otherTile = 'black'
        tilesToFlip = []
        for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x, y = loc_y, loc_x
            x += x_direction
            y += y_direction
            if self.is_on_board(x, y) and self.board[x][y] == otherTile:
                x += x_direction
                y += y_direction
                if not self.is_on_board(x, y):
                    continue
                # 一直走到出界或者不是对方棋子的位置
                while self.board[x][y] == otherTile:
                    x += x_direction
                    y += y_direction
                    if not self.is_on_board(x, y):
                        break
                # 出界了 则没有棋子要反转 OXXXXXXX
                if not self.is_on_board(x, y):
                    continue
                # 是自己的棋子 OXXXXXXO
                if self.board[x][y] == tile:
                    while True:
                        x -= x_direction
                        y -= y_direction
                        # 回到起点则结束
                        if x == loc_y and y == loc_x:
                            break
                        # 需要翻转的棋子
                        tilesToFlip.append([x, y])
        # 将前面临时放上去的棋子去掉，还原棋盘
        self.board[loc_y][loc_x] = 'none'
        # 没有要被翻转的棋子， 则该位置非法
        if len(tilesToFlip) == 0:
            return False
        return tilesToFlip

    def get_valid_moves(self, tile: str):
        """Given tile (black or white), return its available moves"""
        validMoves = []
        for x in range(8):
            for y in range(8):
                if self.is_valid_move(tile, x, y):
                    validMoves.append([x, y])
        return validMoves

    def get_black_white_count(self):
        """获取棋盘上黑白双方的棋子数"""
        black_count = 0
        white_count = 0
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == 'black':
                    black_count += 1
                if self.board[x][y] == 'white':
                    white_count += 1
        return {'black': black_count, 'white': white_count}

    def will_pass(self):
        if not self.is_game_over() and self.next_state_avail_moves_loc == []:
            self.next_state_avail_moves_loc = [-1]
            self.next_state_avail_moves_id = [-1]
            return True
        return False

    def is_game_over(self):
        if self.passed == [True, True]:
            return True
        if len(self.black_id_list) == 0 or len(self.white_id_list) == 0:
            return True
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == 'none':
                    return False
        return True

    # self.board 信息 --> 黑棋id列表和白棋id列表
    def update_black_white_tiles(self):
        black_id_list = []
        white_id_list = []
        for row in range(8):
            for col in range(8):
                tile_id = row * 8 + col
                if self.board[row][col] == 'black':
                    black_id_list.append(tile_id)
                if self.board[row][col] == 'white':
                    white_id_list.append(tile_id)
        self.black_id_list = black_id_list
        self.white_id_list = white_id_list

    # 获取当前棋局局面的特征平面
    def get_feature_planes(self):
        # 返回当前棋盘状态的?个不同的特征平面（? depth）用于输入神经网络
        feature_planes = np.full((config.FEATURE_PLANE_NUM, self.board_size, self.board_size), 0.0)  # 定义4个平面(三维)
        # For plane 1
        black_loc_list = self.moves_2_locations(self.black_id_list)
        black_y_list = [black_loc[0] for black_loc in black_loc_list]
        black_x_list = [black_loc[1] for black_loc in black_loc_list]
        # For plane 2
        white_loc_list = self.moves_2_locations(self.white_id_list)
        white_y_list = [white_loc[0] for white_loc in white_loc_list]
        white_x_list = [white_loc[1] for white_loc in white_loc_list]
        # For plane 3
        non_occupied_moves = np.array(self.get_non_occupied_moves())
        non_occupied_locs = self.moves_2_locations(non_occupied_moves)
        non_occupied_loc_y_list = [non_occupied_loc[0] for non_occupied_loc in non_occupied_locs]
        non_occupied_loc_x_list = [non_occupied_loc[1] for non_occupied_loc in non_occupied_locs]
        """ Plane design """
        # Plane 1: Black stones are 1.0, others 0.0.
        feature_planes[0][black_y_list, black_x_list] = 1.0
        # Plane 2: White stones are 1.0, others 0.0.
        feature_planes[1][white_y_list, white_x_list] = 1.0
        # Plane 3: White and black are 0.0, other non-occupied stones (or cells) are 1.0
        feature_planes[2][non_occupied_loc_y_list, non_occupied_loc_x_list] = 1.0
        # Plane 3: Full ones if current player is black
        feature_planes[3] = 1.0 if self.is_current_player_black() else 0.0
        return feature_planes

    def flip(self, tile, flips, loc_y, loc_x):
        self.board[loc_y][loc_x] = tile
        for x, y in flips:  # 需要翻转的棋子进行变色
            self.board[x][y] = tile

    # perform a move
    def move(self, move, flips=None):
        tile = 'black' if self.is_current_player_black() else 'white'  # for flipping
        other_tile = 'white' if self.is_current_player_black() else 'black'  # for getting the available moves of the next state
        self.all_player_id_list.append(self.current_player_id)
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        self.current_player_is_black = False if self.current_player_is_black else True
        if move == -1:
            if self.passed == [False, False]:
                self.passed = [True, False]
            elif self.passed == [True, False]:
                self.passed = [True, True]
            self.next_state_avail_moves_loc = self.get_valid_moves(other_tile)
            self.next_state_avail_moves_id = self.locations_2_moves(self.next_state_avail_moves_loc)
            will_double_pass = self.will_pass()
            if self.passed == [True, True]:
                self.next_state_avail_moves_loc = []
                self.next_state_avail_moves_id = []
            # self.print_game_information(move)
            return will_double_pass
        self.passed = [False, False]
        self.non_occupied_stones.remove(move)
        self.occupied_stones.append(move)
        loc_y, loc_x = self.move_2_location(move)  # loc_y represents row, while loc_x represents col.
        if flips is not None:
            self.flip(tile, flips, loc_y, loc_x)
        else:
            flips = self.is_valid_move(tile, loc_y, loc_x)
            if flips:
                self.flip(tile, flips, loc_y, loc_x)
            else:
                print('Error occured in false move!')
                return
        self.update_black_white_tiles()
        self.next_state_avail_moves_loc = self.get_valid_moves(other_tile)
        self.next_state_avail_moves_id = self.locations_2_moves(self.next_state_avail_moves_loc)
        will_pass_flag = self.will_pass()  # not game over and no next available moves.
        # self.print_game_information(move)
        return will_pass_flag

    # 黑白棋判断输赢
    def get_game_status(self):
        if not self.is_game_over():
            return -1
        else:
            who_win = None
            scoreBlack = self.get_black_white_count()['black']
            scoreWhite = self.get_black_white_count()['white']
            if scoreBlack > scoreWhite:
                who_win = 1
            if scoreWhite > scoreBlack:
                who_win = 2
            if scoreBlack == scoreWhite:
                who_win = 3
            return who_win

    # 获取当前将要落子的玩家id
    def get_current_player_id(self) -> int:
        return self.current_player_id

    def is_current_player_black(self):
        return self.current_player_is_black

    def get_occupied_stones(self):
        """
        Get occupied moves in which the initial four stones (or tiles) are in consideration
        """
        return self.occupied_stones

    def get_non_occupied_moves(self):
        return self.non_occupied_stones

    def get_history_moves(self):
        """
        Get history moves in which the initial four stones (or tiles) are not in consideration
        """
        return self.get_occupied_stones()[4:]

    def get_all_player_id_list(self):
        return self.all_player_id_list

    # 获得倒数第n步的落子序号
    def get_last_move_id(self, n=1):
        return self.occupied_stones[-n] if n <= len(self.occupied_stones) else -1

    def get_available_moves(self):
        return self.next_state_avail_moves_id

    # Move Id to location (x, y)
    def move_2_location(self, move_id):
        y = move_id // self.board_size
        x = move_id % self.board_size
        return y, x

    # [[0,1], [1,1], ..., [], ...] --> [1, 9, ..., x, ...]
    def locations_2_moves(self, location_list):
        move_list = []
        for location in location_list:
            move = location[0] * self.board_size + location[1]  # row * 8 + col
            move_list.append(move)
        return move_list

    # [1, 9, ..., x, ...] --> [[0,1], [1,1], ..., [], ...]
    def moves_2_locations(self, move_list):
        location_list = []
        for move_id in move_list:
            y, x = self.move_2_location(move_id)
            location_list.append((y, x))
        return location_list

    def print_game_information(self, move='Initial'):
        current_player = 'White' if self.current_player_id == 1 else 'Black'
        # print(self.get_feature_planes()[3])
        print('{} Move {} was performed, the resulting state information is as follow: '.format(current_player, move))
        print('Player Ids: ', self.all_player_id_list)
        print('Occupied: ', self.occupied_stones)
        print('Non-Occupied: ', self.non_occupied_stones)
        print('Black: ', self.black_id_list)
        print('White: ', self.white_id_list)
        print('Next_moves: ', self.get_available_moves())
        print('Current player is black: ', self.is_current_player_black())
        print('History move: ', self.get_history_moves())
        print('Status {}, CurrentId {}'.format(self.get_game_status(), self.get_current_player_id()))
        print('Game over: ', self.is_game_over())
        print()

    @staticmethod
    def is_on_board(x, y):
        return 0 <= x <= 7 and 0 <= y <= 7
