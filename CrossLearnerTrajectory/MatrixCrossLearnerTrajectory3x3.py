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
num_train_episodes = int(1e4)         # Number of episodes for training the players. (for learning)
delta = 0.001
pay_off_tensor_RockPaperScissors = np.array([             # The pay-off matrix
    [[0,-5,10],  # Player 1
     [5,0,-1],
     [-10,1,0]],
    [[0,5,-10],  # Player 2
     [-5,0,1],
     [10,-1,0]]])            

## Set up the game
# Normalize the pay-off tensor (needed for cross learning)
pay_off_tensor = pay_off_tensor_RockPaperScissors
pay_off_tensor = (pay_off_tensor-np.min(pay_off_tensor))/(np.max(pay_off_tensor)-np.min(pay_off_tensor))
game_type = pyspiel.GameType(
    "RPS",
    "RPS",
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
    ["R","P","S"],  # row_action_names    
    ["R","P","S"],  # col_action_names
    list(pay_off_tensor)[0],  # row player utilities
    list(pay_off_tensor)[1]  # col player utilities
)


## Set up the environment (cfr a state of the game, but more elaborate)
env = rl_environment.Environment(game)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]


## Get the pay-off tensor
payoff_tensor = utils.game_payoffs_array(game)

## Set up the replicator dynamics
dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)
  
## Set up the plot
fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(111,projection="3x3")
ax.set_title("Rock-Paper-Scissors")
## Plot the vector field
ax.quiver(dyn)
ax.set_labels(["R","P","S"])

probs1 = [0.15,0.85]
probs2 = [0.2,0.8]

probab = [[[0.75,0.15,1],[0.75,0.15,1]] ] #, [[0.15,0.85],[0.85,0.15]],[[0.85,0.15],[0.2,0.80]],[[0.5,0.5],[0.5,0.5]]]
probab_prisonDilemma = [[[0.85,0.15],[0.15,0.85]], [[0.15,0.85],[0.85,0.15]],[[0.65,0.35],[0.4,0.6]],[[0.35,0.65],[0.65,0.35]],[[0.85,0.15],[0.85,0.15]]]



for prob in probab:

    ## Set up the players: Cross-learning agents
    agents = [CrossLearner(num_actions, player_id = 0, probs = prob[0], delta=delta),
            CrossLearner(num_actions, player_id = 1, probs = prob[1], delta=delta)]

    print("Initial probs for players are: {} and {}.".format(agents[0].getProbs(), agents[1].getProbs()))

    ## Store the probabilities of each episode (needed for the trajectory plot)
    probabilities = np.zeros(( num_train_episodes+1,3))
    probabilities[0,:] = agents[0].getProbs()

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
            
            
        probabilities[cur_episode + 1,: ]= agents[0].getProbs()
        print(agents[0].getProbs())

    ax.plot(probabilities,color="red",alpha=0.5,linewidth=3)
    points = np.zeros(( 1,3))
    points[0,:] = [1/16,10/16,5/16]
    print(points[0,:])
    ax.scatter(points, color = "green")

plt.show()
