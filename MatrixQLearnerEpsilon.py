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
step_size = .01

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
payoff_tensor = pay_off_tensor_battle_of_the_sexes
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
    payoff_tensor[0],  # row player utilities
    payoff_tensor[1]  # col player utilities
)


##  Set up the correct format for the epsilon.
epsilon_schedule = rl_tools.LinearSchedule(0.3,0.05,num_train_episodes)

## Set up the environment (cfr a state of the game, but more elaborate)
env = rl_environment.Environment(game)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]









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


## Prisoners dilemma
""" paretoPoints = np.zeros(( 2,3))
paretoPoints[0,:] = [1,0,1]
paretoPoints[1,:] = [1,1,0]
ax.scatter(paretoPoints[0,:], paretoPoints[1,:], s=300, color = "green")
nash = np.zeros(( 2,1))
nash[0,:] = [0]
nash[1,:] = [0]
ax.scatter(nash[0,:], nash[1,:], s=100, marker = "d", color = "orange") """


## Dispersion game
""" paretoPoints = np.zeros(( 2,2))
paretoPoints[0,:] = [1,0]
paretoPoints[1,:] = [0,1]
ax.scatter(paretoPoints[0,:], paretoPoints[1,:], s=300, color = "green")
nash = np.zeros(( 2,3))
nash[:,:2] = paretoPoints
nash[:,2] = [.5,.5]
ax.scatter(nash[0,:], nash[1,:], s=100, marker = "d", color = "orange") """













Startpoints = [[{0: 0, 1: 0},{0: 0, 1: 0}]]#,[{0: -.01, 1: 0},{0: .015, 1: 0}], [{0: 0, 1: 0.01}, {0: 0, 1: 0.0075}], [{0: .0051, 1: 0.005},{0: .01, 1: .01}]]

for Qs in Startpoints:

    print(Qs)

    ## Set up the players: Qepsilon-greedy Q-learners
    agents = [tabular_qlearner.QLearner(
        player_id=idx,
        num_actions=num_actions,
        epsilon_schedule=epsilon_schedule,
        step_size=step_size
    ) for idx in range(num_players)]

    ## different Q values 
    for i in range(len(agents)):
        agents[i]._q_values['[0.0]']  = Qs[i]

    ## A matrix with the probabilities of each episode to plot in the end (the trajectory)
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

        # Episode is done
        # Let each player learn from the outcome of the episode.
        for agent in agents:
            agent.step(time_step)
                    

    ## Set up the plot
    ax.plot(probabilities[0,:], probabilities[1,:],color="red",alpha=0.5,linewidth=3)
    ax.scatter(probabilities[0,0], probabilities[1,0],color="red",alpha=0.5)



plt.show()
