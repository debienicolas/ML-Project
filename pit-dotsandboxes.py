import os
import numpy as np
import Arena
from open_spiel.python.algorithms.mcts import MCTSBot
from open_spiel.python.bots.uniform_random import UniformRandomBot
from utils import dotdict
import pyspiel
from GNNet import GNNetWrapper as gnn
from GNNEvaluator import GNNEvaluator
from tqdm import tqdm



args = dotdict({
    'numIters': 100,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOf'
    'Queue': 200000,            # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': 'checkpoint',         
    'load_model': True,
    'load_folder_file': ('checkpoint','best.h5'),
    'numItersForTrainExamplesHistory': 20,
})

num_rows = 4
num_cols = 4
game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
game = pyspiel.load_game(game_string)


# hp1 = HumanDotsAndBoxesPlayer(g).play
# hp2 = HumanDotsAndBoxesPlayer(g).play

# rp1 = RandomPlayer(g).play
# rp2 = RandomPlayer(g).play

# grp1 = GreedyRandomPlayer(g).play
# grp2 = GreedyRandomPlayer(g).play


# loading the trained model
n1 = gnn()
n1.load_checkpoint("", "/Users/nicolasdebie/Documents/KU Leuven Burgie/Master 1 fase 2/ML project/ML-Project/checkpoint/best.h5")
n1p = MCTSBot(game,args.cpuct,args.numMCTSSims,
              GNNEvaluator(n1,args))


# loading the random bot
rp2 = UniformRandomBot(1,np.random)

# loading a mcts bot
#mp2 = MCTSBot(game,args.cpuct,args.numMCTSSims,Random)

# Play AlphaZero versus Human
p1 = n1p
p2 = rp2
arena = Arena.Arena(p1,p2, game)

oneWon, twoWon, draws = 0,0,0
rewards = [0.0,0.0]
for i in tqdm(range(20), desc="Playing_games_1"):
    reward = arena.playGame()
    if reward[0] == 1.0:
        oneWon += 1
    elif reward[1] == 1.0:
        twoWon += 1
    else:
        draws += 1

p2 = UniformRandomBot(0,np.random)

arena = Arena.Arena(p2,p1, game)

for i in tqdm(range(20), desc="Playing_games_2"):
    reward = arena.playGame()
    if reward[0] == 1.0:
        twoWon += 1
    elif reward[1] == 1.0:
        oneWon += 1
    else:
        draws += 1

print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))
