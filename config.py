"""(Hyper)paramters Setting"""

# (1) Reward
# Custom reward
REWARD_CUSTOM_OPTIONS = False
BLACK_WIN_SCORE = 1.0
BlACK_LOSE_SCORE = -1.5
WHITE_WIN_SCORE = 1.5
WHITE_LOSE_SCORE = -1.0
# Default reward
NORMAL_SCORE = 1.0

# (2) MCTS
# Dirichlet noise
ADD_DIRICHLET_FOR_EXPANSION = False  # add dirichlet noise when expansion step if true
DIRICHLET_ALPHA = 0.3
DIRICHLET_WEIGHT = 0.25
# UCT
CPUCT = 5 # coefficient C_puct used for UCT when selection step
IS_ALTERNATIVE_TEMPERATURE = False # temperature decrease/increase over time
FIRST_STEP_NUM = 8  # determine what step does it take to start letting temperature decay
FEATURE_PLANE_NUM = 4 # the number of feature plane that represents a state of game

# (3) Training
USE_GPU = False  # use gpu or not
TRAIN_BOARD_SIZE = 8  # specify the size of the board we train
L2_NORM = 1e-4  # L2 penalty
LEARNING_RATE = 1e-3  # learning rate
TRAIN_MCTS_PLYAOUT_NUM = 400  # the number of playout time; that is, how many times MCTS is called for a decision making
DATASET_SIZE = 30000  # size of dataset buffer
BATCH_SIZE = 512
DATASET_SIZE_UPPER_LIMIT = 1000  # start training when a certain figure for data is reached
EPOCHS = 5
SAVE_MODEL_FRENQUENCY = 20  # Save model when a certain figure for the game data collected is reached
SELFPLAY_NUM = 100000  # Perform ? times self-play to collect data
TRAIN_WHICH_NET = 'cnn'  # 'cnn' is short for classic convolutional neural network, while resnet is short for Residual network
SAVE_LATEST_MODEL_PATH = 'model/cnn/othello_8x8/latest.pt'  # the path the latest model is saved
SAVE_GOOD_MODEL_PATH = 'model/cnn/othello_8x8/optimal.pt'  # the path the optimal model is saved
EXISTING_MODEL_PATH = SAVE_LATEST_MODEL_PATH  # Training based on previous model
VISUAL_DATA_PATH = 'visual_data/cnn/'  # use tensorboard for recording loss and observe training process (Deprecated)

# (4) Evaluation
RUN_EVAL = True  # evaluate the latest model if True when training
EVAL_MODEL_FRENQUENCY = 200  # start evaluating when a specific figure for self-play times is reached
EVAL_MCTS_PLAYOUT_NUM = 400  # specify the number of the playout of MCTS of models to be evaluated 
EVAL_NUM = 10  # the number of times two models will play against each other
EVAL_WIN_RATE_THRESHOLD = 0.6  # if reach a specific winning threshold, replace the optimal model with the latest one

# (5) Network
# ResNet
RESNET_FILTER_NUM = 256  # units
RESNET_BLOCKS_NUM = 1  # blocks

# (6) Tk_GUI
GUI_BOARD_GRID = 9
GUI_BOARD_SIZE = GUI_BOARD_GRID - 1

# (7) AI vs Human mode
AI_MCTS_PLAYOUT_NUM = 200  # playout times
AI_NET_TYPE = 'resnet'  # cnn or resnet
AI_RESNET_MODEL_PATH = 'model/resnet/optimal.pt'  # model path
AI_CNN_MODEL_PATH = 'model/cnn/optimal.pt'  # model path
