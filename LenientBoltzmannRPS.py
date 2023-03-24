import numpy as np
import pyspiel
from open_spiel.python.algorithms.boltzmann_tabular_qlearner import BoltzmannQLearner
from open_spiel.python import rl_environment
import open_spiel.python.egt.visualization
import open_spiel.python.egt.dynamics as dynamics
import open_spiel.python.egt.utils as utils
from open_spiel.python import rl_tools
import matplotlib.pyplot as plt
import numpy as np
import collections



## Set up the parameters
kappa = 10
num_train_episodes = int(100000)         # Number of episodes for training the players. (for learning)
pay_off_tensor = np.array([             # The pay-off matrix
    [[0,-5,10],  # Player 1
     [5,0,-1],
     [-10,1,0]],
    [[0,5,-10],  # Player 2
     [-5,0,1],
     [10,-1,0]]])    

pay_off_tensor = np.array([             # The pay-off matrix
    [[0,-.25,.5],  # Player 1
     [.25,0,-.05],
     [-.5,.05,0]],
    [[0,.25,-.5],  # Player 2
     [-.25,0,.05],
     [.05,-.05,0]]])     


game_type = pyspiel.GameType(
    "MatrixGame",
    "MatrixGame",
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
temperature_schedule = rl_tools.LinearSchedule(0.2, 0.001, num_train_episodes)

agents = [BoltzmannQLearner(player_id=idx, num_actions=num_actions,temperature_schedule=temperature_schedule,step_size=0.0001)
          for idx in range(num_players)]


# different Q values 
# for i in range(num_players):
#     agents[i]._q_values = collections.defaultdict()

## Set up the buffers for the leniency.
cache = np.empty((num_players,kappa),dtype=np.int8)
timesteps = [0]*kappa
probabilities = np.zeros((num_players,num_train_episodes//kappa-1))


## Train the agents
# For each episode, do:
for cur_episode in range(num_train_episodes):
    index = 0
    
    # Get the initial state of the game.
    time_step = env.reset()
    
    # Each agent should choose an action 
    agent_output = [agents[player_id].step(time_step, is_evaluation=False) for player_id in range(num_players)]
    
    # Try kappa times to exectute the action in order to find the highest reward.
    while (index<kappa):
        time_step = env.step([x.action for x in agent_output])
        timesteps[index] = time_step
        cache[:,index] = [time_step.rewards[player_id] for player_id in range(num_players)]
        time_step = env.reset()
        index += 1
        

    # Add the probabilities of the actions to the tensor in order to plot them in the end.
    probabilities[:,cur_episode//kappa-1] = [agent_output[player_id].probs[0] for player_id in range(num_players)]

    # Let the players learn from the highest reward.
    for player_id in range(num_players):
        time_step = timesteps[np.argmax(cache[player_id,:])]
        agents[player_id].step(time_step)
    
## The learning is done

print(probabilities)

## Get the pay-off tensor
payoff_tensor = utils.game_payoffs_array(game)

## Set up the replicator dynamics
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
  
## Set up the plot
fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(111,projection="2x2")
ax.set_title("Prisoners Dilemma")
ax.set_xlabel("Player 1")
ax.set_ylabel("Player 2")

## Plot the vector field
ax.quiver(dyn)
ax.plot(probabilities[0,:], probabilities[1,:],color="blue",linewidth=2)
plt.show()