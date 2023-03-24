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
num_train_episodes = int(1*1e6)         # Number of episodes for training the players. (for learning)

pay_off_tensor_battle_of_the_sexes = np.array([            
    [[3,0],  # Player 1
     [0,2]],  
    [[2,0],  # Player 2
     [0,3]]])     

pay_off_tensor_prisoners_dilemma = np.array([            
    [[-1,-4],  # Player 1
     [0,-3]],  
    [[-1,0],  # Player 2
     [-4,-3]]])    

pay_off_tensor_dispersion_game= np.array([             # The pay-off matrix
    [[-1,1],  # Player 1
     [1,-1]],  
    [[-1,1],  # Player 2
     [1,-1]]])

pay_off_tensor_RockPaperScissors = np.array([             # The pay-off matrix
    [[0,-5,10],  # Player 1
     [5,0,-1],
     [-10,1,0]],
    [[0,5,-10],  # Player 2
     [-5,0,1],
     [10,-1,0]]])            

## Set up the game
# Normalize the pay-off tensor (needed for cross learning)
pay_off_tensor = pay_off_tensor_battle_of_the_sexes
pay_off_tensor = (pay_off_tensor-np.min(pay_off_tensor))/(np.max(pay_off_tensor)-np.min(pay_off_tensor))
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
# game = pyspiel.MatrixGame(
#     game_type,
#     {},  # game_parameters
#     ["R","P","S"],  # row_action_names
#     ["R","P","S"],  # col_action_names
#     list(pay_off_tensor)[0],  # row player utilities
#     list(pay_off_tensor)[1]  # col player utilities
# )
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


## Set up the players: Cross-learning agents
print(pay_off_tensor)
'''agents = [CrossLearner(
    num_actions,
    idx,
    None,
    True,
    .001
) for idx in range(num_players)]'''


## Get the pay-off tensor
payoff_tensor = utils.game_payoffs_array(game)

## Set up the replicator dynamics
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
  
## Set up the plot
fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(111,projection="2x2")
ax.set_title("Battle of the Sexes")
ax.set_xlabel("Player 1")
ax.set_ylabel("Player 2")
plt.set_cmap("plasma")
## Plot the vector field
ax.quiver(dyn)


probs1 = [0.15,0.85]
probs2 = [0.2,0.8]

probab = [[[0.15,0.85],[0.8,0.2]], [[0.25,0.75],[0.85,0.15]],[[0.85,0.15],[0.2,0.80]],[[0.8,0.2],[0.15,0.85]]]
delta = 0.0001


for prob in probab:

    agents = [CrossLearner(num_actions, player_id = 0, probs = prob[0], delta=delta),
            CrossLearner(num_actions, player_id = 0, probs = prob[1], delta=delta)]

    # TODO delete statement:
    print("Initial probs for players are: {} and {}.".format(agents[0].getProbs(), agents[1].getProbs()))

    ## Store the probabilities of each episode (needed for the trajectory plot)
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
            agent_output = [agents[player_id].step(time_step, is_evaluation=False) for player_id in range(num_players)]
            # Do the chosen actions and get the new state.
            time_step = env.step([x.action for x in agent_output])
            # TODO delete statement:
            # print("Chosen actions and rewards: {} and {}".format([x.action for x in agent_output], time_step.rewards))

        # Episode is done
        # Let each player learn from the outcome of the episode.
        for agent in agents:
            agent.step(time_step)
            
        # TODO delete statement:
        # print("New probs for players are: {} and {}.".format(agents[0].getProbs(), agents[1].getProbs()))
            
        probabilities[:,cur_episode + 1] = [agent.getProbs(0) for agent in agents]

    ax.plot(probabilities[0,:], probabilities[1,:],color="red",alpha=0.5,linewidth=3)

plt.show()
