import config
from mcts import MCTSPlayer
from game import Game
from gui import GUI
from network import resnet, convnet

if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>GUI界面<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # 初始化棋盘数据(棋盘大小, 输赢判断等)
    game = Game(board_size=config.GUI_BOARD_SIZE)
    # 初始化游戏数据(gui界面实现,人机对弈,自我训练等)
    gui = GUI(game)
    # 创建神经网络用于辅助AI的mcts的部分搜索过程（二选一）
    if config.AI_NET_TYPE == 'cnn':
        net = convnet.NetFunction(config.TRAIN_BOARD_SIZE, model_path=config.AI_CNN_MODEL_PATH)  # 普通cnn网络
    elif config.AI_NET_TYPE == 'resnet':
        net = resnet.NetFunction(config.TRAIN_BOARD_SIZE, model_path=config.AI_RESNET_MODEL_PATH)  # resnet网络
    else:
        net = None
    # 创建AI
    mcts_player = MCTSPlayer(net.get_policy_value_for_mcts, playout_num=config.AI_MCTS_PLAYOUT_NUM)
    # 开启GUI画面
    gui.start_game(mcts_player)

