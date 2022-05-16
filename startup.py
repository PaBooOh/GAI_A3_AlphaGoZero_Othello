import config
from mcts import MCTSPlayer
from game import Game
from gui import GUI
from network import resnet, convnet

if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>GUI界面<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Init game
    game = Game(board_size=config.GUI_BOARD_SIZE)
    # Init gui and bring the core of the game to it
    gui = GUI(game)
    # Build neural network for MCTS for human vs AI mode
    if config.AI_NET_TYPE == 'cnn':
        net = convnet.NetFunction(config.TRAIN_BOARD_SIZE, model_path=config.AI_CNN_MODEL_PATH)  # Classic convolution network
    elif config.AI_NET_TYPE == 'resnet':
        net = resnet.NetFunction(config.TRAIN_BOARD_SIZE, model_path=config.AI_RESNET_MODEL_PATH)  # Residual network
    else:
        net = None
    # Create a AI player based on MCTS and nn
    mcts_player = MCTSPlayer(net.get_policy_value_for_mcts, playout_num=config.AI_MCTS_PLAYOUT_NUM)
    # Open GUI
    gui.start_game(mcts_player)

