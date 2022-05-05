# -*- coding: utf-8 -*-S
from typing import Tuple
import numpy as np
import copy
import config
from basics1 import Foundation
from typing import Optional, Union, Dict


def softmax_func(x):
    if len(x.shape) > 1: # if x is a matrix 
        tmp_max = np.max(x, axis=1)  
        x = x - tmp_max.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp_max = np.sum(x, axis=1)
        x = x / tmp_max.reshape((x.shape[0], 1))
    else:  # if x is an vector
        tmp_max = np.max(x)
        x = x - tmp_max
        x = np.exp(x)
        tmp_max = np.sum(x)
        x = x / tmp_max
    return x


class Node():
    def __init__(self, parent_node, probability):
        self.parent: Optional[Node] = parent_node
        self.children: Dict[int, Node] = {}  # format: {moveId: node}
        self.visit_num = 0 # the number of visiting a node
        self.q_value = 0  # Q value of a node, where it is calculated by w_value / visit_num
        self.w_value = 0 # W value of node, where W is the cumulative leaf values.
        self.prob = probability  # Prior probability of node to be selected

    # Selection
    def select(self) -> Tuple[int, 'Node']:
        """
        Select a child among children according to UCB which computed by Q + U
        """
        selected_node = max(self.children.items(), key=lambda child: child[1].get_ucb(config.CPUCT))
        return selected_node

    # Expansion
    def expand(self, moves_probs, add_dirichlet):
        """
        We here addtionally introduce a special expansion in which dirichlet noise is leveraged for more exploration.
        The equation looks like:  P(s, a) = (1 - epsilon) * p_a + + epsilon * η_a, where epsilon is the dirichlet weight,
        p_a is the prior probability of selecting child node a, and η is the dirichlet noises.

        To slightly change the prior probability, we firstly draw samples from the dirichlet distribution. The shape
        of the result returned is the same as the number of the children we would like to expand.
        Then according to the P(s, a) equation above, prior prob for each move is modified with dirichlet noise.

        The variable of p_a is defined as 'prob'
        The variable of epsilon is defined as 'DIRICHLET_WEIGHT'
        The variable of η is computed by DIRICHLET_ALPHA * np.ones(expanded_nodes_num)
        Variable moves_probs returns [(move_1, prior_prob_1), ..., (move_n, prior_prob_n) ...]
        """
        moves_probs = list(moves_probs)
        if add_dirichlet: # expansion with dirichlet noise
            expanded_nodes_num = len(moves_probs)
            distribution = config.DIRICHLET_ALPHA * np.ones(expanded_nodes_num)
            dirichlet_noises = np.random.dirichlet(distribution)
            for i in range(expanded_nodes_num):
                move = moves_probs[i][0]
                prob = moves_probs[i][1]
                if move not in self.children:
                    epsilon = config.DIRICHLET_WEIGHT
                    eta = dirichlet_noises[i]
                    prob_with_dirichlet_noise = (1 - epsilon) * prob + epsilon * eta
                    self.children[move] = Node(self, prob_with_dirichlet_noise)
        else: # normal expasion without applying dirichlet noise
            for move, prob in moves_probs:
                if move not in self.children:
                    self.children[move] = Node(self, prob)
    
    def simulate(self, copy_game_env, model):
        """
        The current state (represented as the game env) is brought to the neural model.
        It outputs an vector i.e., the prior probabilites of the children; and a scalar i.e., the state value of the leaf node.
        """
        expanded_nodes_prob, leaf_node_value = model(copy_game_env)  # Return the prior probability of each expanded (list), and a value ranging from -1 to 1 (float)
        return expanded_nodes_prob, leaf_node_value


    # Back-propagation
    def backup(self, leaf_node_value):
        # Update the leaf node (No need to update its expanded children)
        self.visit_num += 1
        self.w_value += leaf_node_value
        self.q_value = 1.0 * self.w_value / self.visit_num
        # Update all the parents (include the root node) of the leaf node
        i = 0
        node = self.parent
        while node:
            node.visit_num += 1
            node.w_value += -leaf_node_value if i % 2 == 0 else leaf_node_value
            node.q_value = 1.0 * node.w_value / node.visit_num
            node = node.parent
            i += 1

    def get_ucb(self, c_puct):
        u_value = (c_puct * self.prob * np.sqrt(self.parent.visit_num) / (1 + self.visit_num))
        return self.q_value + u_value

    def is_leaf_node(self):
        return self.children == {}

    def is_root_node(self):
        return self.parent is None


class MCTSPlayer():
    def __init__(self, neural_network, playout_num=400, is_selfplay_mode=False):
        self.root = Node(None, 1.0)  # (moveId, prior probability)
        self.is_selfplay_mode = is_selfplay_mode
        self.model = neural_network  # neural network that takes as input the current state and outputs vector p and scalar v
        self.playout_num = playout_num  # how many the number of playout is performed before a real action is taken

    # Perform an playout including selection, expansion, simulation and backup
    def playout(self, copy_game_env: Foundation):
        current_node = self.root # initialize node
        """
        >>>> Selection
        MCTS takes as input the current state and playout starts, traversing from the roor node (or root state)
        until a leaf node is encoutered. The selection is based on the UCB policy.
        After each selection, we get to the next node. But we also need to accordingly move the 
        corresponding action on the copy game board to reach the next state (Keep in sync).
        """
        while True:
            if current_node.is_leaf_node():
                break
            move, current_node = current_node.select()  # repeatedly select nodes according to UCB (i.e., Q + U) until a leaf node is reached
            copy_game_env.game_move(move)  # expand the search tree
        game_status = copy_game_env.getGameStatus() # Check if the game is over
        """
        If game is not over, expansion and simulation would be applied.
        If game is over, we directly assign a state value to the leaf node which is also the terminal node. Expansion and simulation would not be applied.

        >>>> Expansion and simulation
        Once a leaf node is encoutered, we expand all available children of it at once.
        Then we bring the state stored in the leaf node into neural network.
        This neural network outputs a prior probability vector in which each probability would be assinged to the expanded nodes,
        and a state value (scalar) which would be assinged to the leaf node.
        Here in practice, we first perform simulation to get the state value of leaf node and the prior probabilities of leaf node's children.
        Then we assign state value to leaf node.
        Finally, expand leaf node's children and assign the prior probabilities to the them (i.e., children).
        """
        # If game is not over
        if game_status == -1:
            # Simulation
            expanded_nodes_prob, leaf_node_value = current_node.simulate(copy_game_env, self.model)  # Return the prior probability of each expanded (list), and a value ranging from -1 to 1 (float)
            current_node.expand(expanded_nodes_prob, add_dirichlet=config.ADD_DIRICHLET_FOR_EXPANSION)
        # If game is over
        else:
            if game_status == 3:  # If draw
                leaf_node_value = 0.0
            else:
                # If we design custom reward mechanism
                if config.REWARD_CUSTOM_OPTIONS:
                    # If black wins
                    if copy_game_env.isCurrentPlayerBlack() is False:
                        leaf_node_value = config.BLACK_WIN_SCORE if game_status == copy_game_env.getCurrentPlayerId() else config.WHITE_LOSE_SCORE
                    # If white wins
                    elif copy_game_env.isCurrentPlayerBlack() is True:
                        leaf_node_value = config.WHITE_WIN_SCORE if game_status == copy_game_env.getCurrentPlayerId() else config.BlACK_LOSE_SCORE
                # Default reward mechanism
                else:
                    leaf_node_value = 1.0 if game_status == copy_game_env.getCurrentPlayerId() else -1.0
        """
        >>>> Back-propagation
        There are two scenarios requiring backup:
        (1) All available children are expanded.
        (2) The terminal node is selected in the selection step
        """
        current_node.backup(-leaf_node_value)

    # Get the number of visiting the children of root
    def get_move_visit(self):
        move_list, visit_list = [], []
        for move, node in self.root.children.items():
            move_list.append(move)
            visit_list.append(node.visit_num)
        return move_list, visit_list

    def rebuild_search_tree(self, last_move=None):
        """
        If human vs AI, search tree is totally removed,
        while if AI vs AI for self-play, search tree is partially removed.
        """
        # AI vs AI: Self-play for generating data
        if last_move:
            self.root = self.root.children[last_move]
            self.root.parent = None
        # Human vs AI
        else:
            self.root = Node(None, 1.0)
    
    def perform_mcts(self, game):
        """
        To choose a real move, MCTS combined with neural network would be applied.
        Given a 'playout_num' (Type: Integer), we perform MCTS with NN 'playout_num' times.
        """
        for _ in range(self.playout_num):
            copy_game = copy.deepcopy(game)  # A new copy game environment for mcts is needed
            self.playout(copy_game)  # perform mcts one time
        """
        Then the IDs of the all available moves, which are the root's children, are returned.
        For each moves, its count of visiting time is returned as well.
        """
        move_list, visit_list = self.get_move_visit()
        return move_list, visit_list # Return ID of moves and visiting counts of moves.

    def choose_move(self, game):
        label_pi = np.zeros(game.board_size ** 2)  # PI used as label for training
        move_list, visit_list = self.perform_mcts(game)
        # (1) Mode: AI vs AI (self-play)
        if self.is_selfplay_mode:
            # >> (1.1) temperature decrease over time-step
            if config.IS_ALTERNATIVE_TEMPERATURE:
                # In the first n steps, τ is set to 1 for exploration
                if game.board_size ** 2 - len(game.avail_move_list) <= config.FIRST_STEP_NUM:
                    temperature = 1.0
                else:
                    temperature = 1e-3
                move_probs = softmax_func(1.0 / temperature * np.log(np.array(visit_list) + 1e-10))
                real_move = np.random.choice(move_list, p=move_probs)
                self.rebuild_search_tree(real_move)  # abandon old tree and rebuild a new tree based on the real move
                # unavailable actions is supposed to be 0, while others is based on the softmax.
                label_pi[list(move_list)] = softmax_func(1.0 / 1.0 * np.log(np.array(visit_list) + 1e-10))
                return real_move, label_pi
            # (2.2) temperature is fixed but dirichlet is introduced
            else:
                temperature = 1.0
                move_probs = softmax_func(1.0 / temperature * np.log(np.array(visit_list) + 1e-10))
                # Move with dirichlet noise
                real_move = np.random.choice(move_list, p=(1 - config.DIRICHLET_WEIGHT) * move_probs + config.DIRICHLET_WEIGHT * np.random.dirichlet(
                    config.DIRICHLET_ALPHA * np.ones(len(move_probs))))
                label_pi[list(move_list)] = move_probs
                self.rebuild_search_tree(real_move) # Build new tree based on the real move selected
                return real_move, label_pi
        # (2) Mode: Human vs AI / Mode: Evaluation
        else:
            temperature = 1e-3
            move_probs = softmax_func(1.0 / temperature * np.log(np.array(visit_list) + 1e-10))
            real_move = np.random.choice(move_list, p=move_probs)
            label_pi[list(move_list)] = move_probs
            self.rebuild_search_tree() # Build new tree after the previous one is abandoned completely
            return real_move
