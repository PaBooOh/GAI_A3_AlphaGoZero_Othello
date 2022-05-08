# 参数设置
# 自对弈胜负奖惩回报
REWARD_CUSTOM_OPTIONS = False
BLACK_WIN_SCORE = 1.0
BlACK_LOSE_SCORE = -1.5
WHITE_WIN_SCORE = 1.5
WHITE_LOSE_SCORE = -1.0
NORMAL_SCORE = 1.0

# MCTS
ADD_DIRICHLET_FOR_EXPANSION = False  # 添加狄利克雷噪声在扩展过程的节点上
DIRICHLET_ALPHA = 0.3
DIRICHLET_WEIGHT = 0.25
CPUCT = 5  # UCB算法:控制搜索程度
IS_ALTERNATIVE_TEMPERATURE = False
FIRST_STEP_NUM = 8  # 与temperature系数结合,决定mcts搜索完毕后选择一步棋的探索度
FEATURE_PLANE_NUM = 4

# 训练模型
USE_GPU = False  # 训练是否使用gpu
TRAIN_BOARD_SIZE = 8  # 训练的棋盘尺寸
L2_NORM = 1e-4  # 优化器的l2正则
LEARNING_RATE = 1e-3  # 学习率
TRAIN_MCTS_PLYAOUT_NUM = 400  # mcts搜索次数
DATASET_SIZE = 30000  # 数据集大小
BATCH_SIZE = 512  # 批量抽取的样本(棋局局面)大小
DATASET_SIZE_UPPER_LIMIT = 1000  # 数据集需容量达到一定数目才能开始训练
EPOCHS = 5  # 训练完整数据集的次数
SAVE_MODEL_FRENQUENCY = 20  # 保存模型的频数
SELFPLAY_NUM = 100000  # 自对弈(收集数据)局数
TRAIN_WHICH_NET = 'cnn'  # cnn/resnet
SAVE_LATEST_MODEL_PATH = 'model/cnn/othello_8x8/latest.pt'  # 当前模型的参数保存路径
SAVE_GOOD_MODEL_PATH = 'model/cnn/othello_8x8/optimal.pt'  # 最好模型的参数保存路径
EXISTING_MODEL_PATH = SAVE_LATEST_MODEL_PATH  # 继续训练已有的模型
VISUAL_DATA_PATH = 'visual_data/cnn/8x8/test'  # tensorboard. 存放数据用于可视化

# 评估模型
RUN_EVAL = True  # 是否需要评估模型
EVAL_MODEL_FRENQUENCY = 200  # 评估模型的频数(须大于保存模型的频数)
EVAL_MCTS_PLAYOUT_NUM = 400  # 模型评估的MCTS搜索次数
EVAL_NUM = 10  # 评估局数
EVAL_WIN_RATE_THRESHOLD = 0.6  # 至少赢4盘/平5局/输1局

# ResNet
RESNET_FILTER_NUM = 256  # FILTER数/深度
RESNET_BLOCKS_NUM = 1  # BLOCK数

# Tk_GUI
GUI_BOARD_GRID = 9
GUI_BOARD_SIZE = GUI_BOARD_GRID - 1
GUI_BOARD_VICTORY_NUM = 5

# 人机对战 AI配置
AI_MCTS_PLAYOUT_NUM = 200  # 电脑的搜索次数
AI_NET_TYPE = 'cnn'  # cnn or resnet
AI_RESNET_MODEL_PATH = 'model/resnet/'  # 人机对战时,电脑使用的resnet模型路径
AI_CNN_MODEL_PATH = 'model/cnn/othello_8x8/latest.pt'  # 人机对战时,电脑使用的cnn模型路径
