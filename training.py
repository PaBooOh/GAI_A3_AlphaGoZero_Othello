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

# TensorBoard可视化
# tensorboard --logdir=D:\graduation_project\GobangZero\visual_data\64f_2b_res_8size --port 1235
# writer = SummaryWriter(config.VISUAL_DATA_PATH)


class TrainModel:
    def __init__(self, size, model_path=None, net_type=None):
        self.board_size = size
        self.game = Game(board_size=config.TRAIN_BOARD_SIZE)  # 先定义一个棋盘
        # 存放数据集
        self.data_cache = deque(maxlen=config.DATASET_SIZE)  # 创建FIFO的数据集(队列) 一个state占8个size
        if net_type == 'resnet':
            self.net_func = resnet.NetFunction(self.board_size, model_path=model_path)
        elif net_type == 'cnn':
            self.net_func = convnet.NetFunction(self.board_size, model_path=model_path)
        else:
            print("请指定要训练的网络类型!")
            self.net_func = None
        self.mcts_player = MCTSPlayer(self.net_func.get_policy_value_for_mcts, playout_num=config.TRAIN_MCTS_PLYAOUT_NUM, is_selfplay_mode=True)

    # 自我对弈->收集对弈数据->把对弈数据用来扩充数据集
    def collect_data(self):
        # data -> [([states1],[pi_list1],[z1]), ([states2],[pi_list2],[z2]),...]
        data = self.self_play(self.mcts_player)
        one_game_data = copy.deepcopy(data)
        # 数据增强
        one_game_data = self.expand_data(one_game_data, self.board_size)  # [(局数0的翻转数据1),(局数0的翻转数据1),(局数0的翻转数据1)...]
        # data_cache[0/1/2/3] index代表一局游戏的所有数据 [(局数0的数据),(局数0的翻转数据),(局数1的数据)]
        self.data_cache.extend(one_game_data)
        return len(data), len(self.data_cache)

    # 模型评估对战
    def model_play(self, latest_obj1, good_obj2, round_num):
        # 轮流设置先后手
        who_black, who_white = ['player1', 'player2'] if round_num % 2 == 0 else ['player2', 'player1']  # 交替先后手
        # 初始化棋盘信息
        self.game.initialize_board_info(who_black)
        # 两个模型对弈
        while True:
            mcts_player = latest_obj1 if who_black == 'player1' else good_obj2  # 玩家交替
            move_id = mcts_player.choose_move(self.game)  # mcts搜索
            self.game.move(move_id)  # 更新棋盘信息
            # 每落子一次就check一次
            result_id = self.game.get_game_status()
            # 1=latest 2=good 3=tie
            if result_id in (1, 2, 3):
                return result_id

    # 自我对弈：生成一局棋谱-->状态集S、走子概率集Pi、价值集Z-->数据增强
    def self_play(self, mcts_player: MCTSPlayer):
        # 每开一局都要初始化棋盘信息
        self.game.initialize_board_info()
        # S、Pi、Player_id
        states_list = []  # 一局游戏里的所有棋局状态集
        mcts_pi_list = []  # 一局游戏里的所有局面对应的每个落子位置的概率分布pi(由温度参数给出)
        # 电脑自我对弈
        while True:
            # 当前棋局局面送入mcts+nn 而后输出具体落子位置和落子概率
            move_id, pi = mcts_player.choose_move(self.game)
            # 下一步棋就存起来 (pi.reshape 与state相反)
            states_list.append(self.game.get_feature_planes())  # 每个ndarr类型的局面的4个特征平面(3d)存入?_list
            mcts_pi_list.append(pi)  # 通过多次搜索（playout）后由softmax+tau得出扩展的子节点及其选择概率
            # 更新棋盘数组 记录落子
            self.game.move(move_id)
            # 每下一步check一次胜负
            status = self.game.get_game_status()
            # 分出胜负： 1=player1 2=player2 3=tie
            if status in (1, 2, 3):
                # 下完一局后，记录每个状态下的z值
                move_count = len(self.game.get_all_player_id_list())  # including pass (probably more than one)
                z_list = np.zeros(move_count)  # 一局一共走了多少步, 创建步数记录列表z_list
                # 不是平局:赋对应的胜负reward; 平局:赋全0
                if status != 3:
                    # 强化学习的奖惩原理, 主要考虑到无禁手的不平衡性, 可以自定义reward
                    if config.REWARD_CUSTOM_OPTIONS:
                        # If black wins
                        if not self.game.is_current_player_black():
                            # player_id必须用np.array()处理,否则只会赋-1.0,而1.0不会被赋
                            z_list[np.array(
                                self.game.get_all_player_id_list()) == status] = config.BLACK_WIN_SCORE  # 黑棋赢了
                            z_list[np.array(
                                self.game.get_all_player_id_list()) != status] = config.WHITE_LOSE_SCORE  # 白棋输了
                        # If white wins
                        elif self.game.is_current_player_black():
                            z_list[np.array(
                                self.game.get_all_player_id_list()) == status] = config.WHITE_WIN_SCORE  # 白棋赢了
                            z_list[np.array(
                                self.game.get_all_player_id_list()) != status] = config.BlACK_LOSE_SCORE  # 黑棋输了要扣更多分
                    else:  # 依据Zero围棋的reward
                        z_list[np.array(self.game.get_all_player_id_list()) == status] = config.NORMAL_SCORE
                        z_list[np.array(self.game.get_all_player_id_list()) != status] = -config.NORMAL_SCORE
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
                origin_one_state_data = (
                origin_rot_state_planes, np.flipud(origin_rot_pi).flatten(), value_z)  # pi多转了一次。。
                data_extension.append(origin_one_state_data)
            # （2）镜像
            mirror_state_planes = np.array([np.fliplr(one_plane) for one_plane in one_state])
            mirror_pi = np.fliplr(np.flipud(np.reshape(mcts_pi, (board_size, board_size))))  # 先取反与state一致
            # 镜像翻转4次
            for i in range(1, 5):
                mirror_rot_state_planes = np.array([np.rot90(one_plane, k=i) for one_plane in mirror_state_planes])
                mirror_rot_pi = np.rot90(mirror_pi, k=i)
                mirror_one_state_data = (
                mirror_rot_state_planes, np.flipud(mirror_rot_pi).flatten(), value_z)  # pi多转了一次。。
                data_extension.append(mirror_one_state_data)
        return data_extension

    # 评估模型
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
            print("无法评估模型!请先指定要训练的网络类型!")
            return
        # 计数
        latest_model_win_count, latest_model_tie_count, latest_model_lose_count = 0, 0, 0
        # 评估eval_num局
        for i in range(config.EVAL_NUM):
            winner_id = self.model_play(latest_mcts_player, good_mcts_player, round_num=i)
            if winner_id == 1:
                latest_model_win_count += 1
            elif winner_id == 2:
                latest_model_lose_count += 1
            elif winner_id == 3:
                latest_model_tie_count += 1
        # 胜利+1、平局+0.5
        latest_mcts_win_rate = 1.0 * (latest_model_win_count + 0.5 * latest_model_tie_count) / config.EVAL_NUM
        print("模型MCTS的搜索次数:{}, 最新模型的训练结果-->胜利: {}, 失败: {}, 平局:{}, 分值:{}".format(config.EVAL_MCTS_PLAYOUT_NUM, latest_model_win_count, latest_model_lose_count, latest_model_tie_count, latest_mcts_win_rate))
        return latest_mcts_win_rate

    def start_training(self):
        for game_num in range(1, config.SELFPLAY_NUM):  # 持续几轮
            # 1) 自我对弈收集数据
            start_time = time.time()  # 计时
            episode_len, data_cache_len = self.collect_data()  # 1) 对弈一局->收集该局数据->数据增强->存入数据集
            end_time = time.time()
            cost_time = end_time-start_time
            print("自我对弈第{}局, 该局落子总数:{}, 数据集大小:{}, 该局对弈时间:{}秒".format(game_num, episode_len, data_cache_len, cost_time))
            # writer.add_scalar('steps_per_game', episode_len, game_num)  # tensorboard
            # 2) 训练模型
            if len(self.data_cache) > config.DATASET_SIZE_UPPER_LIMIT:   # 数据集装多大才开始训练
                aggregate_loss, mse_loss, cross_entropy_loss = self.net_func.training(self.data_cache)
                print("aggregate_loss:{}, " "mse_loss:{}, " "cross_entropy_loss:{}".format(aggregate_loss, mse_loss, cross_entropy_loss))
                # writer.add_scalar('aggregate_loss', aggregate_loss, game_num)  # tensorboard
                # writer.add_scalar('cross_entropy_loss', cross_entropy_loss, game_num)  # tensorboard
            # 保存临时模型用于评估（每次训练开始就保存）
            if config.RUN_EVAL and os.path.exists(config.SAVE_GOOD_MODEL_PATH) is False:
                self.net_func.save_model(config.SAVE_GOOD_MODEL_PATH)  # 第一次训练就保存good_model
            # 保存当前模型
            if game_num % config.SAVE_MODEL_FRENQUENCY == 0:
                self.net_func.save_model(config.SAVE_LATEST_MODEL_PATH)
                print(">>>已保存当前模型")
            # 3) 评估当前模型
            if game_num % config.EVAL_MODEL_FRENQUENCY == 0 and config.RUN_EVAL and os.path.exists(config.SAVE_GOOD_MODEL_PATH):
                print(">>>开始对当前模型进行评估...")
                win_ratio = self.model_evaluate(latest_path=config.SAVE_LATEST_MODEL_PATH, good_path=config.SAVE_GOOD_MODEL_PATH)
                if win_ratio >= config.EVAL_WIN_RATE_THRESHOLD:
                    self.net_func.save_model(config.SAVE_GOOD_MODEL_PATH)  # 覆盖之前的good_model
                    print(">>>当前模型强度增强明显 已保存为最好模型")
                else:
                    print(">>>当前模型强度增强不明显")


# 开始训练
if __name__ == '__main__':
    training_process = TrainModel(size=config.TRAIN_BOARD_SIZE, model_path=config.EXISTING_MODEL_PATH, net_type=config.TRAIN_WHICH_NET)
    training_process.start_training()
