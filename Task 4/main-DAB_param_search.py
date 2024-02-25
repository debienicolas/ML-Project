import logging
import coloredlogs
from Coach import Coach
from utils import dotdict
from GNNet import GNNetWrapper as gnn
import pyspiel
import pandas as pd

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO') 

args = dotdict({
    'numIters': 5,
    'numEps': 5,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 20,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 80000,            # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.5,
    'checkpoint': 'checkpoint_param_search',         
    'load_model': False,
    'load_examples': False,
    'load_folder_file': ('checkpoint_256','checkpoint_3.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'saveResults': False,
    'resultsFilePath': 'resultsIteration.csv',

    # GNNet args
    'lr': 0.0001,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 256,
    'l2_coeff': 1e-4
})

# hyperparameters to test
lr = [0.1, 0.01, 0.001, 0.0001]
batch_size = [32, 64]
num_channels = [256, 512]


results = []

results.append({
    'lr': "Learning rate",
    'batch_size': "Batch size",
    'num_channels': "Number of channels",
    'winning_rate_random': 'Winning rate against random',
    'winning_rate_mcts': 'Winning rate against MCTS'
})

for learning_rate in lr:
    for batch in batch_size:
        for channels in num_channels:
            print(f"Learning rate: {learning_rate}, batch size: {batch}, channels: {channels}")
            args.lr = learning_rate
            args.batch_size = batch
            args.num_channels = channels
                
            num_rows = 3
            num_cols = 3
            game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
            game = pyspiel.load_game(game_string)

            gnnet = gnn(args,save_info=False)
                
            c = Coach(game, gnnet, args)
            winningRandom, winningMCTS = c.learn()

            results.append({
                'lr': learning_rate,
                'batch_size': batch,
                'num_channels': channels,
                'winning_rate_random': winningRandom,
                'winning_rate_mcts': winningMCTS
            })


df = pd.DataFrame(results)

df.to_csv("grid_search_results.csv", index=False)