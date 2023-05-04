import open_spiel
from open_spiel.python.algorithms.mcts import Evaluator
import Graph
import numpy as np

class GNNEvaluator(Evaluator):
    def __init__(self, nnet, args):
        self.nnet = nnet
        self.args = args

    def evaluate(self, state):
        """Returns evaluation on given state."""
        _,v = self.nnet.predict(state)
        #print("Evaluation: ", np.array([v,-v]))
        return np.array([v,-v])
        

    def prior(self, state):
        """Returns equal probability for all actions."""
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            pi, _ = self.nnet.predict(state=state)
            # the pi received from predict is a list of probabilities for each action
            pi = Graph.edges_to_actions(state,pi.tolist())
            #print("Pi: ", pi)
            assert len(Graph.edges_to_actions(state,pi)) == state.num_distinct_actions()
            # Still need to filter out the valid actions
            result = [(action,pi[action]) for action,val in enumerate(state.legal_actions_mask()) if val]
            # filter out not none probabilities
            #result = list(filter(lambda x: x[1] != None, result))
            #print("Prior: ", result)
            return result
