import numpy as np
import random

from open_spiel.python import rl_environment

class CrossLearner:
    def __init__(self, numActions: int, randomize: bool = False, delta: float = 1) -> None:
        self._numActions = numActions
        self._delta = delta
        if randomize:
            self._probs = [random.random() for i in range(self._numActions)]
            self._probs = CrossLearner.normalize(self._probs)
        else:
            self._probs = np.ones(self._numActions)/(self._numActions)

    def getProbs(self, action: None|int= None) -> float:
        if (action == None):
            return self._probs
        else:
            return self._probs[action]
        
    def generateAction(self) -> int:
        return np.random.choice(
            np.arange(0, self._numActions), 
            p=self.getProbs())
    
    def setProb(self, action: int, value: float) -> None:
        self._probs[action] = value

    def addToProb(self, action: int, value: float) -> None:
        self.setProb(action, self.getProbs(action)+value)
        
    def updateProbs(self, action: int, reward: float) -> None:
        for i in range(self._numActions):
            if (i==action):
                self.addToProb(i, self._delta * reward*(1-self.getProbs(i)))
            else:
                self.addToProb(i, -self._delta * reward*self.getProbs(i))

    def normalize(x: np.ndarray) -> np.ndarray:
        return x/(np.sum(x))
        #x[-1] = 1-np.sum(x[0:-1])
