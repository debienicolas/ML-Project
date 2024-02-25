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
import csv



args = dotdict({             
    'numMCTSSims': 13,          # Number of games moves for MCTS to simulate.
    'gamesToPlay': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': 'checkpoint_256_update',
    'resultsFilePath': 'scaleExperiment.csv',
})

# save initial parameters
with open(args.resultsFilePath,"a",newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    numMCTS = "numMCTS: " + str(args.numMCTSSims)
    arenaCompare = "arenaCompare: " + str(args.gamesToPlay)
    cpuct = "cpuct: " + str(args.cpuct)
    csv_writer.writerow([numMCTS,arenaCompare,cpuct])

    # write the row titles
    csv_writer.writerow(["size","MCTSWon","MCSTLost","MCTSDraw","MctsWinRate",
                         "RandomWon","RandomLost","RandomDraw","RandomWinRate"])



for i in range(4,15):
    num_rows = i
    num_cols = i

    game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
    game = pyspiel.load_game(game_string)

    # load the trained agent
    n1 = gnn()
    n1.load_checkpoint(args.checkpoint, "best.h5")
    n1p = MCTSBot(game,args.cpuct,13,GNNEvaluator(n1,args))
    
    # loading a mcts bot
    mp2 = MCTSBot(game,args.cpuct,args.numMCTSSims,np.random)

    p1 = n1p
    p2 = mp2
    arena = Arena.Arena(p1,p2, game)

    oneWonRandom, twoWonRandom, drawRandom = arena.playGamesAgainstRandom(p1,args.gamesToPlay)
    RandomWinRate = round(oneWonRandom/args.gamesToPlay,2)

    oneWonMCTS, twoWonMCTS, drawMCTS = arena.playGames(args.gamesToPlay)
    MCTSWinRate = round(oneWonMCTS/args.gamesToPlay,2)

    with open(args.resultsFilePath,"a",newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([str(i),str(oneWonMCTS),str(twoWonMCTS),str(drawMCTS),str(MCTSWinRate),
                             str(oneWonRandom),str(twoWonRandom),str(drawRandom),str(RandomWinRate)])
        
    
    args.numMCTSSims += 8
    


