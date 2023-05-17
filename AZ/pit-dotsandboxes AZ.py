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
from absl import app
from absl import flags
from open_spiel.python.utils import spawn
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero import model as model_lib
import pyspiel
from pathlib import Path

num_rows = 7
num_cols = 7
game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
flags.DEFINE_string("game", game_string, "Name of the game.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("max_simulations", 25, "How many simulations to run.")
flags.DEFINE_integer("train_batch_size", 8, "Batch size for learning.")
flags.DEFINE_integer("replay_buffer_size", 20000,
                     "How many states to store in the replay buffer.")
flags.DEFINE_integer("replay_buffer_reuse", 3,
                     "How many times to learn from each state.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_float("policy_epsilon", 0.25, "What noise epsilon to use.")
flags.DEFINE_float("policy_alpha", 1, "What dirichlet noise alpha to use.")
flags.DEFINE_float("temperature", 1,
                   "Temperature for final move selection.")
flags.DEFINE_integer("temperature_drop", 15,  # Less than AZ due to short games.
                     "Drop the temperature to 0 after this many moves.")
flags.DEFINE_enum("nn_model", "conv2d", model_lib.Model.valid_model_types,    # Andere optie: resnet
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 128, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 3, "How deep should the network be.")
flags.DEFINE_string("path", './checkpointAZ/', "Where to save checkpoints.")
flags.DEFINE_integer("checkpoint_freq", 1, "Save a checkpoint every N steps.")
flags.DEFINE_integer("actors", 2, "How many actors to run.")
flags.DEFINE_integer("evaluators", 1, "How many evaluators to run.")
flags.DEFINE_integer("evaluation_window", 40,
                     "How many games to average results over.")
flags.DEFINE_integer(
    "eval_levels", 7,
    ("Play evaluation games vs MCTS+Solver, with max_simulations*10^(n/2)"
     " simulations for n in range(eval_levels). Default of 7 means "
     "running mcts with up to 1000 times more simulations."))
flags.DEFINE_integer("max_steps", 0, "How many learn steps before exiting.")
flags.DEFINE_bool("quiet", True, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS

args = dotdict({
    'numIters': 20,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 10,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOf'
    'Queue': 200000,            # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': 'checkpoint',         
    'load_model': True,
    'load_folder_file': ('checkpoint','best.h5'),
    'numItersForTrainExamplesHistory': 20,
})

num_rows = 7
num_cols = 7
game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
game = pyspiel.load_game(game_string)

games_to_play = 20

def main(unused_argv):
    # loading the AZ bot

    def build_model(game):
        return model_lib.Model.build_model(
            "conv2d", game.observation_tensor_shape(), game.num_distinct_actions(),
            nn_width=FLAGS.nn_width, nn_depth=FLAGS.nn_depth, weight_decay=FLAGS.weight_decay, learning_rate=FLAGS.learning_rate, path = FLAGS.path)

    model = build_model(game)
    model.from_checkpoint("checkpoint-61", path=Path('../checkpointAZ/'))
    evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    rp2 = MCTSBot(game,FLAGS.uct_c,FLAGS.max_simulations,
                evaluator)
    print("loaded AZ")
    # hp1 = HumanDotsAndBoxesPlayer(g).play
    # hp2 = HumanDotsAndBoxesPlayer(g).play

    # rp1 = RandomPlayer(g).play
    # rp2 = RandomPlayer(g).play

    # grp1 = GreedyRandomPlayer(g).play
    # grp2 = GreedyRandomPlayer(g).play

    # loading the trained model
    n1 = gnn()
    n1.load_checkpoint("", "checkpoint_4x4/best.h5")
    n1p = MCTSBot(game,FLAGS.uct_c,FLAGS.max_simulations,
                GNNEvaluator(n1,args))
    print("loaded gnn")



    # n2 = gnn()
    # n2.load_checkpoint("", "/Users/nicolasdebie/Documents/KU Leuven Burgie/Master 1 fase 2/ML project/ML-Project/checkpoint/best.h5")
    # n2p = MCTSBot(game,args.cpuct,args.numMCTSSims,
    #                 GNNEvaluator(n2,args))


    # loading a mcts bot
    #mp2 = MCTSBot(game,args.cpuct,args.numMCTSSims,Random)

    # Play AlphaZero versus Human
    p1 = n1p
    p2 = rp2
    arena = Arena.Arena(p1,p2, game)

if __name__ == "__main__":
  print(model_lib.Model.valid_model_types)
  with spawn.main_handler():
    app.run(main)