import logging

from tqdm import tqdm
from open_spiel.python.bots.uniform_random import UniformRandomBot
import numpy as np
from open_spiel.python.algorithms.mcts import MCTSBot
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            The returns for the first player
        """
        players = [self.player1,self.player2]
        state = self.game.new_initial_state()
        # beginning player is 0 (player 1)
        curPlayer = state.current_player()
        it = 0
        while not state.is_terminal():
            curPlayer = state.current_player()
            it += 1

            bot = players[curPlayer]
            action = bot.step(state)

            state.apply_action(action)
        
        return state.rewards() 

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameRewards = self.playGame(verbose=verbose)
            if gameRewards == [1.0, -1.0]:
                oneWon += 1
            elif gameRewards== [-1.0, 1.0]:
                twoWon += 1
            else:
                draws += 1
        
        # switch players so player 2 starts
        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameRewards == [1.0, -1.0]:
                twoWon += 1
            elif gameResult == [-1.0, 1.0]:
                oneWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
    
    def playGamesAgainstRandom(self, player1, num, verbose=False):
        rp2 = UniformRandomBot(1,np.random)
        arena = Arena(player1, rp2, self.game)
        oneWon, twoWon, draws = 0,0,0
        
        for i in tqdm(range(num//2), desc="Playing_games_1"):
            reward = arena.playGame()
            if reward[0] == 1.0:
                oneWon += 1
            elif reward[1] == 1.0:
                twoWon += 1
            else:
                draws += 1
        
        rp2 = UniformRandomBot(0,np.random)
        arena = Arena(rp2,player1, self.game)

        for i in tqdm(range(num//2), desc="Playing_games_2"):
            reward = arena.playGame()
            if reward[0] == 1.0:
                twoWon += 1
            elif reward[1] == 1.0:
                oneWon += 1
            else:
                draws += 1
        print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))
        return oneWon, twoWon, draws
    
    def playGamesAgainstMCTS(self, player1, num, verbose=False):
        evaluator = RandomRolloutEvaluator(1, np.random)
        p2 = MCTSBot(self.game,1,50,evaluator)
        arena = Arena(player1, p2, self.game)
        oneWon, twoWon, draws = 0,0,0
        
        for i in tqdm(range(num//2), desc="Playing_games_1"):
            reward = arena.playGame()
            if reward[0] == 1.0:
                oneWon += 1
            elif reward[1] == 1.0:
                twoWon += 1
            else:
                draws += 1
        
  
        arena = Arena(p2,player1, self.game)

        for i in tqdm(range(num//2), desc="Playing_games_2"):
            reward = arena.playGame()
            if reward[0] == 1.0:
                twoWon += 1
            elif reward[1] == 1.0:
                oneWon += 1
            else:
                draws += 1
        print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))
        return oneWon, twoWon, draws


