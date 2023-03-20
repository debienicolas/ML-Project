import numpy as np
import random
from open_spiel.python import rl_environment

class CrossLearner:
    '''
    An implementation of an agent that uses the cross learning algorithm
    for matrix games. 
    The pay-off matrix has to be min-max normalised, s.t. the rewards satisfy 0 <= reward <= 1.
    
    An example for how to use it:
        agent = CrossLearner(num_actions)
        action = agent.generateAction()
        time_step = env.step(action)
        agent.updateProbs(action, time_step.reward)
    '''
    
    def __init__(self, numActions: int, randomize: bool = False, delta: float = 1) -> None:
        '''
        Constructor for this learner.

        @param {numactions} The number of actions that this agent can take.
        @param {randomize} A boolean indicating whether the actions for this agent have 
            random probabilities (if True) or uniform probabilities (if False).
        @param {delta} The learning rate for the cross learning algorithm. 
        '''
        
        self._numActions = numActions
        self._delta = delta
        if randomize:
            self._probs = [random.random() for i in range(self._numActions)]
            self._probs = CrossLearner._normalize(self._probs)
        else:
            self._probs = np.ones(self._numActions)/(self._numActions)

    def getProbs(self, action: None|int= None) -> float:
        '''
        Return the probabilities.
        If an action is given as a parameter, then the probability for that action is returned.

        @param {action} None to return a list of probabilities or an integer for the action.
        @return The probability of the given action or the list of probabilities if None is given as a parameter.
        '''
        if (action == None):
            return self._probs
        else:
            return self._probs[action]
        
    def generateAction(self) -> int:
        '''
        Return an action according to the current probabilities of the actions for this agent.
        '''
        return np.random.choice(
            np.arange(0, self._numActions), 
            p=self.getProbs())
    
    def _setProb(self, action: int, value: float) -> None:
        '''
        Set the probability for the given action.

        @param {action} An integer specifying the action for which you want to update the probability.
        @param {value} A float indicating the new probability for the given action.
        '''
        self._probs[action] = value

    def _addToProb(self, action: int, value: float) -> None:
        '''
        Add the given value to the current probability for the given action.
        
        @param {action} An integer specifying the action for which you want to increase the probability.
        @param {value} The value to add to the current porbability of the given action.
        '''
        self._setProb(action, self.getProbs(action)+value)
        
    def updateProbs(self, action: int, reward: float) -> None:
        '''
        Update the probability of the given action, based on the reward for taking that action.
        This update is according to the cross learning algorithm.

        @param {action} An integer specifying the action that you have taken.
        @param {reward} A float specifying the reward for taking that action
        '''
        for i in range(self._numActions):
            if (i==action):
                self._addToProb(i, self._delta * reward*(1-self.getProbs(i)))
            else:
                self._addToProb(i, -self._delta * reward*self.getProbs(i))

    def _normalize(x: np.ndarray) -> np.ndarray:
        '''
        Normalize the given vector such that it sums up to 1.
        '''
        return x/(np.sum(x))
        #x[-1] = 1-np.sum(x[0:-1])
