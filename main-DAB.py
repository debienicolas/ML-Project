import logging

import coloredlogs

from Coach import Coach
from utils import dotdict
from GNNet import GNNetWrapper as gnn
import pyspiel


log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 100,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOf'
    'Queue': 80000,            # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': 'checkpoint',         
    'load_model': True,
    'load_folder_file': ('checkpoint','best.h5'),
    'numItersForTrainExamplesHistory': 4,
})


def main():
    log.info('Loading: game')
    # create a game object with board size 7
    num_rows = 4
    num_cols = 4
    game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'

    game = pyspiel.load_game(game_string)

    # log loading the graph neural network
    log.info('Loading the GNNet 🤖')

    gnnet = gnn(game)
    
    if args.load_model:
        # if you want to load the model 
        gnnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')
    log.info('Loading the Coach...')
    
    c = Coach(game, gnnet, args)


    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
