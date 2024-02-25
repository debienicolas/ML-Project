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
log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(args)  # the competitor network
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.winRatesRandom = []
        self.winRatesMCTS = []

        if self.args.saveResults:
            with open(self.args.resultsFilePath,"a",newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Iteration","won_random","lost_random","draw_random","winning rate","new_model", "won_mcts","lost_mcts","draw_mcts","winning rate"])

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        # no need to look at symmetries(rotation and reflection) when using GNN => decreases training time
        trainExamples = []
        #print("Starting an episode...")
        
        state = self.game.new_initial_state()
        self.curPlayer = state.current_player()
        episodeStep = 0

        while not state.is_terminal():

            episodeStep += 1
            temp = int(episodeStep <= self.args.tempThreshold)
            
            self.curPlayer = state.current_player()

            pi,action= self.mcts.step_with_policy_training(state,temp)
            
            trainExamples.append([Graph.state_to_graph_data(state),pi,state.current_player(),None])
            
            state.apply_action(action)
            
        
        # Game is done now
        # returns the state graph, the policy, and the reward for the current player
        reward = state.rewards()[self.curPlayer]
        return [(x[0], x[1], reward * ((-1) ** (x[2] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # change sliding window: Dynamic sliding window , start with a low window, then move up until 20 previous iterations are used
            if self.args.numItersForTrainExamplesHistory < 20 and i > 3:
                self.args.numItersForTrainExamplesHistory += 1
            
            log.info(f'Starting Iter #{i} ...')

            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                # double ended queue, automatically removes the oldest element if the maxlen is reached
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # Perform selfplay and add the examples trainExamples
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(
                        self.game,
                        self.args.cpuct,
                        self.args.numMCTSSims,
                        GNNEvaluator(self.nnet)) 

                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            print("shuffling examples")
            
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            
            # previous MCTS
            pmcts  = MCTSBot(self.game,self.args.cpuct,self.args.numMCTSSimsArena,GNNEvaluator(self.pnet)) 

            # train the GNNet on the new examples
            print("Amount of training examples: ", len(trainExamples))
            self.nnet.train(trainExamples)

            # new MCTS and saving a temporary checkpoint
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            nmcts = MCTSBot(self.game,self.args.cpuct,self.args.numMCTSSimsArena,
                        GNNEvaluator( self.nnet)) 

            # evaluate the new model against the old one
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(pmcts,nmcts, self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

                # model is rejected, don't pit against benchmarks
                new_model = False
                self.winRatesRandom.append(False)
                self.winRatesMCTS.append(False)
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                # model is accepted, pit against benchmarks
                new_model = True
                won,lost,draw = arena.playGamesAgainstRandom(nmcts,20)
                won_m,lost_m,draw_m = arena.playGamesAgainstMCTS(nmcts,20)
                winning_rate = round(won/(won+lost+draw),4)
                winning_rate_m = round(won_m/(won_m+lost_m+draw_m),4)
                self.winRatesRandom.append(winning_rate)
                self.winRatesMCTS.append(winning_rate_m)

            # save the results og the evaulations against benchmarks to a csv file
            if new_model and self.args.saveResults:
                with open(self.args.resultsFilePath,"a",newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([i,won,lost,draw,winning_rate,new_model,won_m,lost_m,draw_m,winning_rate_m])
        return self.winRatesRandom, self.winRatesMCTS

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        # filename = checkpoint/checkpoint_i.pth.tar.examples
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
