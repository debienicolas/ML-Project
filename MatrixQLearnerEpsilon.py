## Based on:     print("The utilities for the players after their first game (before learning) are: {}".format(time_step.rewards))


## Import all necessary modules
import pyspiel
import numpy as np
from open_spiel.python import rl_environment
import open_spiel.python.egt.visualization
import open_spiel.python.egt.dynamics as dynamics
import open_spiel.python.egt.utils as utils
import matplotlib.pyplot as plt
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
import numpy as np




## Set up the parameters
num_train_episodes = int(1000)         # Number of episodes for training the players. (for learning)
epsilon_schedule = .1                   # The epsilon for the epsilon-greedy step.


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


##  Set up the correct format for the epsilon.
epsilon_schedule = rl_tools.LinearSchedule(0.2,0,num_train_episodes//2)
epsilon_schedule = rl_tools.ConstantSchedule(0.1)

## Set up the environment (cfr a state of the game, but more elaborate)
env = rl_environment.Environment(game)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

## Set up the players: Qepsilon-greedy Q-learners
agents = [tabular_qlearner.QLearner(
    player_id=idx,
    num_actions=num_actions,
    epsilon_schedule=epsilon_schedule,
    step_size=0.0001
) for idx in range(num_players)]
# TODO delete statement:


probabilities = np.zeros((num_players, num_train_episodes))


## Train the agents
# For each episode, do:
for cur_episode in range(num_train_episodes):
    # Get the initial state of the game.
    time_step = env.reset()
    # As long as the game has not finished, do:
    while not time_step.last():
        # Each agent should choose an action and learn from the state it is in (time_step)
        agent_output = [agents[player_id].step(time_step, is_evaluation=False) for player_id in range(num_players)]
        probabilities[:,cur_episode] = [agent_output[player_id].probs[0] for player_id in range(num_players)]
        # Do the chosen actions and get the new state.
        time_step = env.step([x.action for x in agent_output])
        # TODO delete statement:
        # print("Chosen actions and rewards: {} and {}".format([x.action for x in agent_output], time_step.rewards))

    # Episode is done
    # Let each player learn from the outcome of the episode.
    for agent in agents:
        agent.step(time_step)
                
print(probabilities)

## Get the pay-off tensor
payoff_tensor = utils.game_payoffs_array(game)

## Set up the replicator dynamics
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
  
## Set up the plot
fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(111,projection="2x2")

## Plot the vector field
ax.quiver(dyn)
ax.plot(probabilities[0,:], probabilities[1,:])
plt.show()
