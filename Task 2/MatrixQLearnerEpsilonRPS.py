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
num_train_episodes = int(1000)        
step_size = .01

pay_off_tensor = np.array([             
    [[0,-.25,.5],  # Player 1
     [.25,0,-.05],
     [-.5,.05,0]],
    [[0,.25,-.5],  # Player 2
     [-.25,0,.05],
     [.5,-.05,0]]])     


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


##  Set up the correct format for the epsilon.
epsilon_schedule = rl_tools.LinearSchedule(0.3,0.05,num_train_episodes)

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
#ax.set_title("Prisoners Dilemma")
ax.set_xlabel("Player 1")
ax.set_ylabel("Player 2")
## Plot the vector field
ax.quiver(dyn)

## Plot the Nash equilibria and Pareto optimality points
paretoPoints = np.zeros(( 3,3))
paretoPoints[0,:] = [1,0,0]
paretoPoints[1,:] = [0,0,1]
paretoPoints[2,:] = [0,1,0]
ax.scatter(paretoPoints, s=300, color = "green")
nash = np.zeros(( 1,3))
nash[0,:] = [1/16,10/16,5/16]
ax.scatter(nash, s=100, marker = "d", color = "orange")
ax.set_labels(["R","P","S"])


# Initialize the different Q-values for the learner.
Startpoints = [[{0: 0, 1: 0,2:0},{0: 0, 1: 0,2:0}]]#,[{0: 0.02, 1: .01,2:.012},{0: 0.02, 1: .01,2:.012}]]
for Qs in Startpoints:
    print(Qs)
    ## Set up the players: Qepsilon-greedy Q-learners
    agents = [tabular_qlearner.QLearner(
        player_id=idx,
        num_actions=num_actions,
        epsilon_schedule=epsilon_schedule,
        step_size=step_size
    ) for idx in range(num_players)]
    for i in range(len(agents)):
        agents[i]._q_values['[0.0]']  = Qs[i]

    ## The probabilities of each episode to plot in the end (the trajectory)
    probabilities = np.zeros((num_train_episodes,3))


    ## Train the agents
    # For each episode, do:
    for cur_episode in range(num_train_episodes):
        # Get the initial state of the game.
        time_step = env.reset()
        # As long as the game has not finished, do:
        while not time_step.last():
            # Each agent should choose an action and learn from the state it is in (time_step)
            agent_output = [agents[player_id].step(time_step, is_evaluation=False) for player_id in range(num_players)]
            probabilities[cur_episode,:] = agent_output[0].probs
            # Do the chosen actions and get the new state.
            time_step = env.step([x.action for x in agent_output])

        # Episode is done
        # Let each player learn from the outcome of the episode.
        for agent in agents:
            agent.step(time_step)
                    

    ## Set up the plot
    ax.plot(probabilities,color="red",alpha=0.5,linewidth=3)
    xxx = np.zeros((1,3))
    xxx[0,:] = probabilities[0]
    ax.scatter(xxx,color="red",alpha=0.5)



plt.show()
