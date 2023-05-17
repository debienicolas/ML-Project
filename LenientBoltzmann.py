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
num_train_episodes = int(10000)         # Number of episodes for training the players. (for learning)
pay_off_tensor_battle_of_the_sexes = np.array([            
    [[3,0],  # Player 1
     [0,2]],  
    [[2,0],  # Player 2
     [0,3]]])   
kappa = 10
step_size = 0.001

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
pay_off_tensor = pay_off_tensor_dispersion_game
#pay_off_tensor = (pay_off_tensor-np.min(pay_off_tensor))/(np.max(pay_off_tensor)-np.min(pay_off_tensor))
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
    ["A","B"],  # row_action_names
    ["A","B"],  # col_action_names
    list(pay_off_tensor)[0],  # row player utilities
    list(pay_off_tensor)[1]  # col player utilities
)

## Set up the environment (cfr a state of the game, but more elaborate)
env = rl_environment.Environment(game)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]
temperature_schedule = rl_tools.LinearSchedule(.3, 0.01, num_train_episodes)


## Get the pay-off tensor
payoff_tensor = utils.game_payoffs_array(game)

## Set up the replicator dynamics
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
  
## Set up the plot
fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(111,projection="2x2")
#ax.set_title("Prisoners Dilemma")
ax.set_xlabel("Player 1")
ax.set_ylabel("Player 2")
## Plot the vector field
ax.quiver(dyn)

## Battle of the sexes
paretoPoints = np.zeros(( 2,2))
paretoPoints[0,:] = [0,1]
paretoPoints[1,:] = [0,1]
ax.scatter(paretoPoints[0,:], paretoPoints[1,:], s=300, color = "green")
nash = np.zeros(( 2,3))
nash[:,:2] = paretoPoints
nash[:,2] = [.6,.4]
ax.scatter(nash[0,:], nash[1,:], s=100, marker = "d", color = "orange")


""" ## Prisoners dilemma
paretoPoints = np.zeros(( 2,3))
paretoPoints[0,:] = [1,0,1]
paretoPoints[1,:] = [1,1,0]
ax.scatter(paretoPoints[0,:], paretoPoints[1,:], s=300, color = "green")
nash = np.zeros(( 2,1))
nash[0,:] = [0]
nash[1,:] = [0]
ax.scatter(nash[0,:], nash[1,:], s=100, marker = "d", color = "orange") """


""" ## Dispersion game
paretoPoints = np.zeros(( 2,2))
paretoPoints[0,:] = [1,0]
paretoPoints[1,:] = [0,1]
ax.scatter(paretoPoints[0,:], paretoPoints[1,:], s=300, color = "green")
nash = np.zeros(( 2,3))
nash[:,:2] = paretoPoints
nash[:,2] = [.5,.5]
ax.scatter(nash[0,:], nash[1,:], s=100, marker = "d", color = "orange")
 """





Startpoints = [[{0: 0, 1: 0},{0: 0, 1: 0}],[{0: -.01, 1: 0},{0: .015, 1: 0}], [{0: 0, 1: 0.01}, {0: 0, 1: 0.0075}], [{0: .02, 1: 0.005},{0: .01, 1: .01}]]

for Qs in Startpoints:

    print(Qs)

    agents = [BoltzmannQLearner(player_id=idx, num_actions=num_actions,temperature_schedule=temperature_schedule,step_size=step_size)
                for idx in range(num_players)]

    ## different Q values 
    for i in range(len(agents)):
        agents[i]._q_values['[0.0]']  = Qs[i]


    ## Set up the buffers for the leniency.
    cache = np.empty((num_players,kappa),dtype=np.int8)
    timesteps = [0]*kappa

    ## Set up the probability tensor to plot in the end
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

    ## Set up the plot
    ax.plot(probabilities[0,:], probabilities[1,:],color="red",alpha=0.5,linewidth=3)
    ax.scatter(probabilities[0,0], probabilities[1,0],color="red",alpha=0.5)
    


plt.show()
