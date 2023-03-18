## Based on:     print("The utilities for the players after their first game (before learning) are: {}".format(time_step.rewards))


## Import all necessary modules
import pyspiel
import numpy as np
from open_spiel.python import rl_environment
from CrossLearner import CrossLearner
import open_spiel.python.egt.visualization
import open_spiel.python.egt.dynamics as dynamics
import open_spiel.python.egt.utils as utils
import matplotlib.pyplot as plt
import numpy as np


## Set up the parameters
num_train_episodes = int(5*1e4)        # Number of episodes for training the players. (for learning)
pay_off_tensor = np.array([
    [[3,0],  # Player 1
     [5,1]],  
    [[3,5],  # Player 2
     [0,1]]])


## Set up the game
pay_off_tensor = (pay_off_tensor-np.min(pay_off_tensor))/(np.max(pay_off_tensor)-np.min(pay_off_tensor))
print(pay_off_tensor)
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
    list(pay_off_tensor)[0],  # row player utilities
    list(pay_off_tensor)[1]  # col player utilities
)


## Set up the environment (cfr a state of the game, but more elaborate)
env = rl_environment.Environment(game)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]


## Set up the players: Qepsilon-greedy Q-learners
agents = [CrossLearner(
    #player_id=idx,
    num_actions,
    True,
    .001
) for idx in range(num_players)]
# TODO delete statement:
print("Initial probs for players are: {} and {}.".format(agents[0].getProbs(), agents[1].getProbs()))

probabilities = np.zeros((num_players, num_train_episodes+1))
probabilities[:,0] = [agent.getProbs(0) for agent in agents]

## Train the agents
# For each episode, do:
for cur_episode in range(num_train_episodes):
    # Get the initial state of the game.
    time_step = env.reset()
    # As long as the game has not finished, do:
    while not time_step.last():
        # Each agent should choose an action and learn from the state it is in (time_step)
        actions = [agents[player_id].generateAction() for player_id in range(num_players)]
        # Do the chosen actions and get the new state.
        time_step = env.step(actions)

    # Episode is done
    # Let each player learn from the outcome of the episode.
    for player_id in range(num_players):
        rewards = time_step.rewards
        agents[player_id].updateProbs(actions[player_id], rewards[player_id])
        #print("The chosen actions are {}, and the rewards are {}".format(actions, rewards))
        
    probabilities[:,cur_episode + 1] = [agent.getProbs(0) for agent in agents]
    # TODO delete statement:
    #print("The probs for players are: {} and {}.".format(agents[0].getProbs(), agents[1].getProbs()))





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
