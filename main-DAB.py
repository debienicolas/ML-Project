import logging
from Coach import Coach
from utils import dotdict
from GNNet import GNNetWrapper as gnn
import pyspiel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

args = dotdict({
    'numIters': 400,                        # Number of total iterations
    'numEps': 50,                           # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 35,                    # Threshold amount of actions until temperature is 0.
    'updateThreshold': 0.55,                # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 80000,                 # Number of game examples to train the neural networks.
    'numMCTSSims': 100,                     # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,                     # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 3.0,                           # Constant in the PUCB formula
    'checkpoint': 'checkpoint_last_attempt',        # location of checkpoint folder 
    'load_model': False ,                            # load the model or not
    'load_examples': False,                         # load the examples or not
    'load_folder_file': ('checkpoint_last_attempt','checkpoint_3.pth.tar'), # location of training examples
    'numItersForTrainExamplesHistory': 5,           # Number of training examples to be used for training before new examples are added.
    'saveResults': True,                            # save the results or not
    'resultsFilePath': 'resultsIteration_last_attempt.csv', # location of the results file

    'numMCTSSimsArena': 15,                # Number of games moves for MCTS to simulate during arena play.

    # GNNet args
    'lr': 0.00008,                        # learning rate
    'epochs': 20,                         # number of epochs
    'batch_size': 64,
    'num_channels': 128,
    'l2_coeff': 1e-4
})


def main():
    log.info('Loading: game')
    
    # game creation
    num_rows = 4
    num_cols = 4
    game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
    game = pyspiel.load_game(game_string)

    # log loading the graph neural network
    log.info('Loading the GNNet 🤖')

    gnnet = gnn(args,save_info=False)
    
    if args.load_model:
        # loads the model
        gnnet.load_checkpoint(args.checkpoint, 'best.h5')
    else:
        log.warning('Not loading a checkpoint!')
    log.info('Loading the Coach...')
    
    # instantiate learning environment
    c = Coach(game, gnnet, args)

    # load the training examples 
    if args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
