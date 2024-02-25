import logging
import coloredlogs
from Coach import Coach
from utils import dotdict
from GNNet import GNNetWrapper as gnn
import pyspiel
import pandas as pd
import csv

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO') 

args = dotdict({
    'numIters': 5,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 20,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 80000,            # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.5,
    'checkpoint': 'checkpoint_param_search',         
    'load_model': False,
    'load_examples': False,
    'load_folder_file': ('checkpoint_256','checkpoint_3.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'saveResults': False,
    'resultsFilePath': 'resultsIteration.csv',

    'numMCTSSimsArena': 15,

    # GNNet args
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
    'l2_coeff': 1e-4
})

# hyperparameters to test
tempThreshold = [10,2030]
cpuct = [1,2,3,4,5]


with open("grid_search_results_exploration.csv","a",newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["temperatureThreshold","cpuct","winning_rate_random","winning_rate_mcts"])

# init the game
num_rows = 4
num_cols = 4
game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
game = pyspiel.load_game(game_string)

for temp in tempThreshold:
    for puct in cpuct:
        print(f"temp: {temp}, cpuct: {puct}")
        args.tempThreshold = temp
        args.cpuct = puct
        
        gnnet = gnn(args,save_info=False)
            
        c = Coach(game, gnnet, args)

        winningRandom, winningMCTS = c.learn()

        with open("grid_search_results_exploration.csv","a",newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([temp,puct,winningRandom,winningMCTS])

