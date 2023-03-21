import pyspiel
import open_spiel.python.egt.visualization
import open_spiel.python.egt.dynamics as dynamics
import open_spiel.python.egt.utils as utils
import matplotlib.pyplot as plt
import numpy as np


## Set up the game
game_type = pyspiel.GameType(
    "battleOfTheSexes",
    "Battle Of The Sexes",
    pyspiel.GameType.Dynamics.SIMULTANEOUS,
    pyspiel.GameType.ChanceMode.DETERMINISTIC,
    pyspiel.GameType.Information.ONE_SHOT,
    #pyspiel.GameType.Utility.ZERO_SUM,
    pyspiel.GameType.Utility.IDENTICAL,
    pyspiel.GameType.RewardModel.TERMINAL,
    2,  # max num players
    2,  # min_num_players
    True,  # provides_information_state
    True,  # provides_information_state_tensor
    False,  # provides_observation
    False,  # provides_observation_tensor
    dict()  # parameter_specification
)
game = pyspiel.MatrixGame(
    game_type,
    {},  # game_parameters
    ["A","B"],  # row_action_names
    ["A","B"],  # col_action_names
    [[-1,1],[1,-1]],  # row player utilities
    [[-1,1],[1,-1]]  # col player utilities
)

## Get the pay-off tensor
payoff_tensor = utils.game_payoffs_array(game)

## Set up the replicator dynamics
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
  
## Set up the plot
fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(111,projection="2x2")

## Plot the vector field
ax.quiver(dyn)
plt.show()