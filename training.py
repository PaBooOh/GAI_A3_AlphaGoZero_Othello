import copy
import numpy as np
import config
import time
import os
import pickle
from gui import GUI
from collections import deque
from game import Game
from mcts import MCTSPlayer
from network import resnet, convnet
from torch.utils.tensorboard import SummaryWriter

# TensorBoard
# tensorboard --logdir=D:\graduation_project\GobangZero\visual_data\64f_2b_res_8size --port 1235
# writer = SummaryWriter(config.VISUAL_DATA_PATH)


class TrainModel:
    def __init__(self, size, model_path=None, net_type=None):
        self.board_size = size
        self.game = Game(board_size=config.TRAIN_BOARD_SIZE)
        self.data_cache = deque(maxlen=config.DATASET_SIZE)  # For storing self-play data. Structure: FIFO, where a state take up 8 size.
        if net_type == 'resnet':
            self.net_func = resnet.NetFunction(self.board_size, model_path=model_path)
        elif net_type == 'cnn':
            self.net_func = convnet.NetFunction(self.board_size, model_path=model_path)
        else:
            print("Please specify a network!")
            self.net_func = None
        self.mcts_player = MCTSPlayer(self.net_func.get_policy_value_for_mcts, playout_num=config.TRAIN_MCTS_PLYAOUT_NUM, is_selfplay_mode=True)

    # Collect game data by self-play. Moreover, the game data generated would be leveraged for data augmentation.
    def collect_data(self):
        # data -> [([states1],[pi_list1],[z1]), ([states2],[pi_list2],[z2]),...]
        data = self.self_play(self.mcts_player)
        one_game_data = copy.deepcopy(data)
        # Data augmentation: transformation, mirror...
        one_game_data = self.expand_data(one_game_data, self.board_size)
        # data_cache[0/1/2/3], where the index represents a game data
        self.data_cache.extend(one_game_data)
        return len(data), len(self.data_cache)

    # Adversarial evaluation between models
    def model_play(self, latest_obj1, good_obj2, round_num):
        # Alternate side per 5 games
        who_black, who_white = ['player1', 'player2'] if round_num % 2 == 0 else ['player2', 'player1']
        self.game.initialize_board_info(who_black) # Init the board and game
        # Pitting two models against each other
        while True:
            mcts_player = latest_obj1 if who_black == 'player1' else good_obj2  # Alternate
            move_id = mcts_player.choose_move(self.game)  # Choose a real move based on MCTS that introduce neural network
            self.game.move(move_id)  # Update board information
            result_id = self.game.get_game_status()  # Check the game results (or status)
            who_black = 'player2' if who_black == 'player1' else 'player1'
            # We set the result like: 1=latest model 2=optimal model 3=draw
            if result_id in (1, 2, 3):
                return result_id

    # Self-play：to get a game data that consists of State list、Pi list and Z list.
    def self_play(self, mcts_player: MCTSPlayer):
        self.game.initialize_board_info()
        # S、Pi、Player_id
        states_list = []  # State list
        mcts_pi_list = []  # Pi list (the probabilities of selecting moves)
        # Self-play
        while True:
            move_id, pi = mcts_player.choose_move(self.game)  # Choose a real move based on MCTS that introduce neural network
            states_list.append(self.game.get_feature_planes())  # Assume the feature planes represent a state
            mcts_pi_list.append(pi)
            self.game.move(move_id)  # Update board information
            status = self.game.get_game_status()  # Check the game results (or status)
            # We set the result like: 1=player1, 2=player2 and 3=draw
            if status in (1, 2, 3):
                move_count = len(self.game.get_all_player_id_list())  # including pass (probably more than one)
                z_list = np.zeros(move_count)
                # if Draw, reward is 0
                if status != 3:
                    # Custom reward
                    if config.REWARD_CUSTOM_OPTIONS:
                        # If black wins
                        if not self.game.is_current_player_black():
                            z_list[np.array(
                                self.game.get_all_player_id_list()) == status] = config.BLACK_WIN_SCORE  # Black wins
                            z_list[np.array(
                                self.game.get_all_player_id_list()) != status] = config.WHITE_LOSE_SCORE  # White loses
                        # If white wins
                        elif self.game.is_current_player_black():
                            z_list[np.array(
                                self.game.get_all_player_id_list()) == status] = config.WHITE_WIN_SCORE  # White wins
                            z_list[np.array(
                                self.game.get_all_player_id_list()) != status] = config.BlACK_LOSE_SCORE  # Black loses
                    else:  # By default
                        z_list[np.array(self.game.get_all_player_id_list()) == status] = config.NORMAL_SCORE
                        z_list[np.array(self.game.get_all_player_id_list()) != status] = -config.NORMAL_SCORE
                # zip as (S、Pi、Z)
                one_game_data = list(zip(states_list, mcts_pi_list, z_list))
                return one_game_data

    # Data augmentation
    def expand_data(self, one_game_data, board_size):
        # format: [(state, pi, z), ..., ...]
        data_extension = []
        # mcts_pi_len: size**2 vector
        for one_state, mcts_pi, value_z in one_game_data:  # loop over (s, pi, z), where the data type are np.arr, np.arr and np.float64, respectively.
            # （1）Flip 4 times
            for i in range(1, 5):
                origin_rot_state_planes = np.array([np.rot90(one_plane, k=i) for one_plane in one_state])
                origin_rot_pi = np.rot90(np.flipud(np.reshape(mcts_pi, (board_size, board_size))), k=i)
                origin_one_state_data = (
                origin_rot_state_planes, np.flipud(origin_rot_pi).flatten(), value_z)
                data_extension.append(origin_one_state_data)
            # （2）Mirror
            mirror_state_planes = np.array([np.fliplr(one_plane) for one_plane in one_state])
            mirror_pi = np.fliplr(np.flipud(np.reshape(mcts_pi, (board_size, board_size))))  # inverse
            for i in range(1, 5):
                mirror_rot_state_planes = np.array([np.rot90(one_plane, k=i) for one_plane in mirror_state_planes])
                mirror_rot_pi = np.rot90(mirror_pi, k=i)
                mirror_one_state_data = (
                mirror_rot_state_planes, np.flipud(mirror_rot_pi).flatten(), value_z)
                data_extension.append(mirror_one_state_data)
        return data_extension

    # Evaluation on model
    def model_evaluate(self, latest_path, good_path):
        if config.TRAIN_WHICH_NET == 'resnet':
            latest_resnet_func = resnet.NetFunction(self.board_size, model_path=latest_path)
            good_resnet_func = resnet.NetFunction(self.board_size, model_path=good_path)
            latest_mcts_player = MCTSPlayer(latest_resnet_func.get_policy_value_for_mcts, playout_num=config.EVAL_MCTS_PLAYOUT_NUM)
            good_mcts_player = MCTSPlayer(good_resnet_func.get_policy_value_for_mcts, playout_num=config.EVAL_MCTS_PLAYOUT_NUM)
        elif config.TRAIN_WHICH_NET == 'cnn':
            latest_convnet_func = convnet.NetFunction(self.board_size, model_path=latest_path)
            good_convnet_func = convnet.NetFunction(self.board_size, model_path=good_path)
            latest_mcts_player = MCTSPlayer(latest_convnet_func.get_policy_value_for_mcts, playout_num=config.EVAL_MCTS_PLAYOUT_NUM)
            good_mcts_player = MCTSPlayer(good_convnet_func.get_policy_value_for_mcts, playout_num=config.EVAL_MCTS_PLAYOUT_NUM)
        else:
            print("Please specify a model for evaluation!")
            return
        # Count the winning rate...
        latest_model_win_count, latest_model_tie_count, latest_model_lose_count = 0, 0, 0
        # Set the number of game for evaluation
        for i in range(config.EVAL_NUM):
            winner_id = self.model_play(latest_mcts_player, good_mcts_player, round_num=i)
            if winner_id == 1:
                latest_model_win_count += 1
            elif winner_id == 2:
                latest_model_lose_count += 1
            elif winner_id == 3:
                latest_model_tie_count += 1
        # Win for +1 and draw for +0.5
        latest_mcts_win_rate = 1.0 * (latest_model_win_count + 0.5 * latest_model_tie_count) / config.EVAL_NUM
        print("Adversarial evaluation result of the latest model-->Win: {}, Lose: {}, Draw:{}, score:{}".format(latest_model_win_count, latest_model_lose_count, latest_model_tie_count, latest_mcts_win_rate))
        return latest_mcts_win_rate

    # Start training
    def start_training(self):
        for game_num in range(1, config.SELFPLAY_NUM):  # Set the number of the playout for choosing a real move (or an action)
            # 1) With self-play, data are collected.
            start_time = time.time()
            episode_len, data_cache_len = self.collect_data()  # 1) Take self-play data and the data augmented to cache
            end_time = time.time()
            cost_time = end_time-start_time
            print("Self-play_nums:{}, Total moves:{}, Size of data cache:{}, Took:{} seconds".format(game_num, episode_len, data_cache_len, cost_time))
            # writer.add_scalar('steps_per_game', episode_len, game_num)  # tensorboard
            # 2) Training
            if len(self.data_cache) > config.DATASET_SIZE_UPPER_LIMIT:   # set a threshold of the data cache to specify when to train
                aggregate_loss, mse_loss, cross_entropy_loss = self.net_func.training(self.data_cache)
                print("Aggregate_loss:{}, " "mse_loss:{}, " "cross_entropy_loss:{}".format(aggregate_loss, mse_loss, cross_entropy_loss))
                # writer.add_scalar('aggregate_loss', aggregate_loss, game_num)  # tensorboard
                # writer.add_scalar('cross_entropy_loss', cross_entropy_loss, game_num)  # tensorboard
            # Save the optimal model if it is None
            if config.RUN_EVAL and os.path.exists(config.SAVE_GOOD_MODEL_PATH) is False:
                self.net_func.save_model(config.SAVE_GOOD_MODEL_PATH)
            # Save the latest model
            if game_num % config.SAVE_MODEL_FRENQUENCY == 0:
                self.net_func.save_model(config.SAVE_LATEST_MODEL_PATH)
                print(">>>Latest model saved!")
            # 3) Evaluate the latest model
            if game_num % config.EVAL_MODEL_FRENQUENCY == 0 and config.RUN_EVAL and os.path.exists(config.SAVE_GOOD_MODEL_PATH):
                print(">>>Start evaluating the latest model ...")
                win_ratio = self.model_evaluate(latest_path=config.SAVE_LATEST_MODEL_PATH, good_path=config.SAVE_GOOD_MODEL_PATH)
                if win_ratio >= config.EVAL_WIN_RATE_THRESHOLD:
                    self.net_func.save_model(config.SAVE_GOOD_MODEL_PATH)  # replacing the optimal one
                    print(">>>The latest model is better than the optimal model. Updated!")
                else:
                    print(">>The latest model is worse than the optimal model. Replacement is canceled.")

    def models_battle(self):
        if config.RUN_EVAL and os.path.exists(
                config.SAVE_GOOD_MODEL_PATH):
            print(">>>Start evaluating the latest model ...")
            win_ratio = self.model_evaluate(latest_path=config.SAVE_LATEST_MODEL_PATH,
                                            good_path=config.SAVE_GOOD_MODEL_PATH)
            print('Win_ratio: ',win_ratio)
            # if win_ratio >= config.EVAL_WIN_RATE_THRESHOLD:
            #     self.net_func.save_model(config.SAVE_GOOD_MODEL_PATH)  # replacing the optimal one
            #     print(">>>The latest model is better than the optimal model. Updated!")
            # else:
            #     print(">>The latest model is worse than the optimal model. Replacement is canceled.")

# Start training
if __name__ == '__main__':
    training_process = TrainModel(size=config.TRAIN_BOARD_SIZE, model_path=config.EXISTING_MODEL_PATH, net_type=config.TRAIN_WHICH_NET)
    # training_process.start_training()
    training_process.models_battle()