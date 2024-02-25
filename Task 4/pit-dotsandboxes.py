import os
import numpy as np
import Arena
from open_spiel.python.algorithms.mcts import MCTSBot
from open_spiel.python.bots.uniform_random import UniformRandomBot
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator
from utils import dotdict
import pyspiel
from GNNet import GNNetWrapper as gnn
from GNNEvaluator import GNNEvaluator
from tqdm import tqdm

args = dotdict({
    'numIters': 100,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 20,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 80000,            # Number of game examples to train the neural networks.
    'numMCTSSims': 14,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.5,
    'checkpoint': 'checkpoint',         
    'load_model': True,
    'load_examples': True,
    'load_folder_file': ('checkpoint','checkpoint_0.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'saveResults': True,
    'resultsFilePath': 'resultsIteration_1.csv',

    # GNNet args
    'lr': 0.001,
    'epochs': 15,
    'batch_size': 32,
    'num_channels': 512,
    'l2_coeff': 1e-4
})

num_rows = 7
num_cols = 7
game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
game = pyspiel.load_game(game_string)




# loading the trained model
n1 = gnn(args)
n1.load_checkpoint("checkpoint", "best.h5")
n1p = MCTSBot(game,args.cpuct,args.numMCTSSims,GNNEvaluator(n1))


# loading the random bot
rp2 = UniformRandomBot(1,np.random)

# loading a mcts bot
evaluator = RandomRolloutEvaluator(1, np.random)
mp2 = MCTSBot(game,1.4,25,evaluator)

# Play AlphaZero versus Human
p1 = n1p
p2 = rp2
arena = Arena.Arena(p1,p2, game)

# play against random
oneWon, twoWon, draws = arena.playGamesAgainstRandom(p1,20)

# play against mcts
oneWon, twoWon, draws = arena.playGamesAgainstMCTS(p1,20)

n2p  = gnn(args)
n2p.load_checkpoint("checkpoint_256_update", "best.h5")
n2p = MCTSBot(game,args.cpuct,args.numMCTSSims,GNNEvaluator(n2p))
arena = Arena.Arena(n1p,n2p,game)

# play against other loaded model 
oneWon, twoWon, draws = arena.playGames(20)
