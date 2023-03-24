"""Boltzmann Q learning agent.

This algorithm is a variation of Q learning that uses action selection
based on boltzmann probability interpretation of Q-values.

For more details, see equation (2) page 2 in
   https://arxiv.org/pdf/1109.1528.pdf
"""

import numpy as np

from open_spiel.python import rl_tools
from open_spiel.python.algorithms import boltzmann_tabular_qlearner


class LenientBoltzmannQLearner(boltzmann_tabular_qlearner.BoltzmannQLearner):
  """Tabular Boltzmann Q-Learning agent.

  See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.

  The tic_tac_toe example uses the standard Qlearner. Using the
  BoltzmannQlearner is
  identical and only differs in the initialization of the agents.
  """

  def __init__(self,
               player_id,
               num_actions,
               step_size=0.1,
               discount_factor=1.0,
               temperature_schedule=rl_tools.ConstantSchedule(.5),
               centralized=False):
    super().__init__(
        player_id,
        num_actions,
        step_size=step_size,
        discount_factor=discount_factor,
        temperature_schedule=temperature_schedule,
        centralized=centralized)
    self._previous_Lenient_action = None
    self._is_Lenient_evaluation = False

  
  def lenient_step(self, time_step, action):
    self._is_Lenient_evaluation = True
    self._previous_Lenient_action = action
    result = self.step(time_step, False)
    self._is_Lenient_evaluation = False
    self._previous_Lenient_action = None
    return result
    
    

  def _get_action_probs(self, info_state, legal_actions, epsilon):
    """Returns a selected action and the probabilities of legal actions."""
    action, probs = super()._get_action_probs(info_state,legal_actions,epsilon)
    if self._is_Lenient_evaluation:
        return self._previous_Lenient_action, probs
    else:
        return action, probs
