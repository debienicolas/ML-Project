import open_spiel
from open_spiel.python.algorithms.mcts import Evaluator
import Graph
import numpy as np


# Evaluator for MCTS made for GNN model
class GNNEvaluator(Evaluator):
    def __init__(self, nnet):
        self.nnet = nnet

    def evaluate(self, state):
        """Returns evaluation on given state."""

        _,v = self.nnet.predict(state)

        return np.array([v,-v])
        

    def prior(self, state):
        """Returns prior probability for all actions."""
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            pi, _ = self.nnet.predict(state=state)
            # the pi received from predict is a list of probabilities for each action

            # we need to convert the edges to the correct order corresponding to the actions
            pi = Graph.edges_to_actions(state,pi.tolist())

            # Now we filter out the valid actions for the given state
            result = [(action,pi[action]) for action,val in enumerate(state.legal_actions_mask()) if val]
            
            return result
