import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from open_spiel.python.algorithms.mcts import MCTSBot
from GNNEvaluator import GNNEvaluator
import Graph
import csv
from MCTS import MCTS
from GNNet import GNNetWrapper as gnn
import pyspiel

num_rows = 3
num_cols = 3
game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
game = pyspiel.load_game(game_string)

# initialize the nnet
nnet = gnn()

# initialize the MCTS
mcts = MCTS(game,1.4,15,GNNEvaluator(nnet)) 


trainExamples = []
state = game.new_initial_state()

while not state.is_terminal():
    
    curPlayer = state.current_player()
    print(state)
    pi,action= mcts.step_with_policy_training(state)
    #pi = Graph.edges_to_actions(state,pi)
    pi = [(i,x) for i,x in enumerate(pi)]
    print("Policy: ", pi)
    print("Action: ", action)

    trainExamples.append([Graph.state_to_graph_data(state),pi,state.current_player(),None])

    state.apply_action(action)
   


# Game is done now
# returns the state graph, the policy, and the reward for the current player
reward = state.rewards()[curPlayer]
#print([(x[0], x[1], reward * ((-1) ** (x[2] != curPlayer))) for x in trainExamples])