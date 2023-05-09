import logging
import coloredlogs
from datetime import datetime

from Coach import Coach
from utils import dotdict
from GNNet import GNNetWrapper as gnn
import pyspiel


import sys
sys.path.insert(0, "/Users/nicolasdebie/Documents/KU Leuven Burgie/Master 1 fase 2/ML project/ML-Project/torch_geometric")

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

now = datetime.now()

args = dotdict({
    'numIters': 20,
    'numEps': 60,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOf'
    'Queue': 200000,            # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': 'checkpoint_4x4_256',         
    'load_model': False,
    'load_examples': False,
    'load_folder_file': ('checkpoint_2x2','checkpoint_4.pth.tar'),
    'numItersForTrainExamplesHistory': 6,

    'numMCTSSimsArena':25,
    'resultsFilePath': 'results4x4' + now.strftime("%m-%d %H:%M") +  '.csv'
    #'resultsFilePath': 'results4x405-09 01:53.csv'
})


def main():
    log.info('Loading: game')
    # create a game object with board size 4
    num_rows = 4
    num_cols = 4
    game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'

    game = pyspiel.load_game(game_string)

    # log loading the graph neural network
    log.info('Loading the GNNet 🤖')

    gnnet = gnn(argsargs=args,save_info=True)
    
    if args.load_model:
        # if you want to load the model 
        gnnet.load_checkpoint(args.checkpoint, 'best.h5')
        #gnnet.load_checkpoint('', '/Users/nicolasdebie/Documents/KU Leuven Burgie/Master 1 fase 2/ML project/ML-Project/checkpoint_6/best.h5')
    else:
        log.warning('Not loading a checkpoint!')
    log.info('Loading the Coach...')
    
    # load training_examples
    c = Coach(game, gnnet, args)
    # load the training examples 
    if args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
