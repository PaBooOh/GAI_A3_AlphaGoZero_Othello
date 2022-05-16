import numpy as np
import config


class Game:
    def __init__(self, board_size):
        self.board_size = board_size  # Board size
        self.non_occupied_stones = list(set([i for i in range(self.board_size ** 2)]) - {27, 28, 35, 36})  # Record the stones not on the board
        self.occupied_stones = [27, 28, 35, 36]  # Record the stones on the board
        self.all_player_id_list = []  # Record the sequence of the two game players
        self.current_player_is_black = None  # Record if the current player represents black side (The current player is the one who is about to put a stone)
        self.board = self.get_new_board()  # 8 x 8 array for storing game state, where a position is in ['black','white','none']
        self.next_state_avail_moves_loc = self.get_valid_moves('black')  # [[2, 3], [3, 2], [4, 5], [5, 4]]
        self.next_state_avail_moves_id = self.locations_2_moves(self.next_state_avail_moves_loc)
        self.black_id_list = [28, 35]
        self.white_id_list = [27, 36]
        self.passed = [False, False]

    def initialize_board_info(self, who_first='player1'):
        self.current_player_id = 1 if who_first == 'player1' else 2  # For Human vs AI mode
        self.non_occupied_stones = list(set([i for i in range(self.board_size ** 2)]) - {27, 28, 35, 36}) # Record the stones not on the board
        self.occupied_stones = [27, 28, 35, 36]  # Record the stones on the board
        self.non_occupied_stones = [i for i in self.non_occupied_stones if i not in self.occupied_stones]
        self.all_player_id_list = []  # Record the sequence of the two game players
        self.board = self.get_new_board() # 8 x 8 array for storing game state, where a position is in ['black','white','none']
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
        # There are four stones initially on the board
        board[3][3] = 'white'
        board[3][4] = 'black'
        board[4][3] = 'black'
        board[4][4] = 'white'
        return board

    def is_valid_move(self, tile: str, loc_y, loc_x):
        # check if the action is valid
        if not self.is_on_board(loc_y, loc_x) or self.board[loc_y][loc_x] != 'none':
            return False
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
                while self.board[x][y] == otherTile:
                    x += x_direction
                    y += y_direction
                    if not self.is_on_board(x, y):
                        break
                if not self.is_on_board(x, y):
                    continue
                if self.board[x][y] == tile:
                    while True:
                        x -= x_direction
                        y -= y_direction
                        if x == loc_y and y == loc_x:
                            break
                        # Store the stones that are needed to be flipped
                        tilesToFlip.append([x, y])
        self.board[loc_y][loc_x] = 'none'
        # This action is valid if there are no stones for flipping
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
        """Get the count of black stone and white stone"""
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
        """After an action is performed, check if the next available action have to be PASS"""
        if not self.is_game_over() and self.next_state_avail_moves_loc == []:
            self.next_state_avail_moves_loc = [-1]
            self.next_state_avail_moves_id = [-1]
            return True
        return False

    def is_game_over(self):
        """Check if is game over"""
        # Double pass
        if self.passed == [True, True]:
            return True
        # Stones on the board are black/white only
        if len(self.black_id_list) == 0 or len(self.white_id_list) == 0:
            return True
        # No more empty cell
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == 'none':
                    return False
        return True

    def update_black_white_tiles(self):
        """Get the black and white stones' distribution on the board"""
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

    def get_feature_planes(self):
        """Get the custom feature planes (or state representation) for neural network"""
        feature_planes = np.full((config.FEATURE_PLANE_NUM, self.board_size, self.board_size), 0.0)  # Define ? feature plane in 3-dims
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
        """Flip the stones"""
        self.board[loc_y][loc_x] = tile
        for x, y in flips:  # 需要翻转的棋子进行变色
            self.board[x][y] = tile

    def move(self, move, flips=None):
        """
        Perform an action (or move) and accordingly update the information about the next state
        Futher, a check that the next action is PASS will be returned.
        """
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

    def get_game_status(self):
        """
        Get the status of the game, in which -1 shows the game is ongoing, while 3 represents the game is draw.
        1 and 2 indicate the winning player id, respectively.
        """
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

    def get_current_player_id(self):
        """Get the current player id, where the current player basically is the one who is about to perform the next action (or move)"""
        return self.current_player_id

    def is_current_player_black(self):
        """Check whether the current player is black or not"""
        return self.current_player_is_black

    def get_occupied_stones(self):
        """Get occupied moves in which the initial four stones (or tiles) are in consideration"""
        return self.occupied_stones

    def get_non_occupied_moves(self):
        """Get empty moves (or cells to be exact)"""
        return self.non_occupied_stones

    def get_history_moves(self):
        """
        Get history moves in which the initial four stones (or tiles) are not in consideration
        """
        return self.get_occupied_stones()[4:]

    def get_all_player_id_list(self):
        """Get the sequence of the two game players in a game completed"""
        return self.all_player_id_list

    def get_last_move_id(self, n=1):
        """Get the last n moves Id for the design of the feature planes in general"""
        return self.occupied_stones[-n] if n <= len(self.occupied_stones) else -1

    def get_available_moves(self):
        """Based on the current state, return the available actions (or moves)"""
        return self.next_state_avail_moves_id

    def move_2_location(self, move_id):
        """Convert move Id to location (x, y)"""
        y = move_id // self.board_size
        x = move_id % self.board_size
        return y, x

    def locations_2_moves(self, location_list):
        """Convert move Ids to locations, e.g., [[0,1], [1,1], ..., [], ...] --> [1, 9, ..., x, ...]"""
        move_list = []
        for location in location_list:
            move = location[0] * self.board_size + location[1]  # row * 8 + col
            move_list.append(move)
        return move_list

    def moves_2_locations(self, move_list):
        """Convert locations to move Ids, e.g., [1, 9, ..., x, ...] --> [[0,1], [1,1], ..., [], ...]"""
        location_list = []
        for move_id in move_list:
            y, x = self.move_2_location(move_id)
            location_list.append((y, x))
        return location_list

    def print_game_information(self, move='Initial'):
        """Print some basic and necessary information for debugging"""
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
        """Check if a move is on the board"""
        return 0 <= x <= 7 and 0 <= y <= 7
