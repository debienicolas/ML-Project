import numpy as np
import pyspiel
from LenientBoltzmannQLearner import LenientBoltzmannQLearner
from open_spiel.python import rl_environment
import open_spiel.python.egt.visualization
import open_spiel.python.egt.dynamics as dynamics
import open_spiel.python.egt.utils as utils
from open_spiel.python import rl_tools
import matplotlib.pyplot as plt
import numpy as np
import collections



## Set up the parameters
num_train_episodes = int(10000)         # Number of episodes for training the players. (for learning)
pay_off_tensor = np.array([             # The pay-off matrix
    [[3,0],  # Player 1
     [0,2]],  
    [[2,0],  # Player 2
     [0,3]]])

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
temperature_schedule = rl_tools.LinearSchedule(0.3, 0.01, num_train_episodes)

agents = [LenientBoltzmannQLearner(player_id=idx, num_actions=num_actions,temperature_schedule=temperature_schedule,step_size=0.001)
          for idx in range(num_players)]


# different Q values 
# for i in range(num_players):
#     agents[i]._q_values = collections.defaultdict()


kappa = 10
cache = np.empty((num_players,kappa,2),dtype=np.int8)
index = 0
probabilities = np.zeros((num_players,num_train_episodes//kappa-1))


## Train the agents
# For each episode, do:
for cur_episode in range(num_train_episodes):
    is_evaluation = True
    #print(index)
    # Get the initial state of the game.
    time_step = env.reset()
    # As long as the game has not finished, do:
    if index % kappa == 0 and cur_episode != 0:
        is_evaluation = False
    while not time_step.last() and is_evaluation:
        # Each agent should choose an action and learn from the state it is in (time_step)
        agent_output = [agents[player_id].step(time_step, is_evaluation=is_evaluation) for player_id in range(num_players)]
        
        # Do the chosen actions and get the new state.

        time_step = env.step([x.action for x in agent_output])
        if is_evaluation:
            for player_id in range(num_players):
                cache[player_id,index] = [agent_output[player_id].action , time_step.rewards[player_id]]
        # TODO delete statement:
        # print("Chosen actions and rewards: {} and {}".format([x.action for x in agent_output], time_step.rewards))
    index += 1
    # Episode is done
    # Let each player learn from the outcome of the episode.
    if not is_evaluation:
        #print(cache)
        actions = [cache[player_id,np.argmax(cache[player_id,:,1]),0] for player_id in range(num_players)]
        agent_output =  [agents[player_id].lenient_step(time_step,actions[player_id]) for player_id in range(num_players)]
        probabilities[:,cur_episode//kappa-1] = [agent_output[player_id].probs[0] for player_id in range(num_players)]
        time_step = env.step([x.action for x in agent_output])

        for agent in agents:
            agent.step(time_step)
        index = 1
    
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

