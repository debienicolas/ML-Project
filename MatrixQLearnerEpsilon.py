## Based on:     print("The utilities for the players after their first game (before learning) are: {}".format(time_step.rewards))


## Import all necessary modules
import pyspiel
import numpy as np
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import rl_tools
from open_spiel.python import rl_environment



## Set up the parameters
num_train_episodes = int(20)        # Number of episodes for training the players. (for learning)
num_eval_episodes = int(1000)       # Number of games to play for calculating the mean utility of a specific player. (for evaluation)
eval_freq = int(1)                  # How frequent (the number of training episodes) an evaluation of the players has to occur.
epsilon_schedule = .1               # The epsilon for the epsilon-greedy step.



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
epsilon_schedule = rl_tools.ConstantSchedule(float(epsilon_schedule))


## Set up the environment (cfr a state of the game, but more elaborate)
env = rl_environment.Environment(game)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

## Set up the players: Qepsilon-greedy Q-learners
agents = [tabular_qlearner.QLearner(
    player_id=idx,
    num_actions=num_actions,
    epsilon_schedule=epsilon_schedule
) for idx in range(num_players)]



def eval_agents(env, agents, num_episodes):
  """
  Auxiliry function for evaluating agents.
  This function runs the game a <num_episodes> amount of times    
  and calculates the rewards of the players after each game.
  This function outputs the mean of these rewards as a numpy array with the mean reward for each player."""
  rewards = np.array([0] * env.num_players, dtype=np.float64)
  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.last():
        agent_output = [agents[player_id].step(time_step, is_evaluation=True) for player_id in range(num_players)]
        time_step = env.step([x.action for x in agent_output])
    rewards += time_step.rewards
  rewards /= num_episodes
  return rewards






## Run the game when the players have not 'learned' yet.
# Get the initial state of the game.
time_step = env.reset() # A time step contains the data of a state of the game.
# As long as the game has not ended, do:
while not time_step.last():
    # Each agent should choose an action and learn from the state it is in (time_step)
    agent_output = [agents[player_id].step(time_step, is_evaluation=True) for player_id in range(num_players)]
    # Do the chosen actions and get the new state.
    time_step = env.step([x.action for x in agent_output])
print("The utilities for the players after their first game (before learning) are: {}".format(time_step.rewards))


## Train the agents
# For each episode, do:
for cur_episode in range(num_train_episodes):
    # Evaluate the current players, if an amount of <eval_freq> episodes has passed.
    if cur_episode%int(eval_freq) == 0:
        # Get the average rewards for each player after playing an amount of <num_eval_episodes> games.
        avg_rewards = eval_agents(env, agents, num_eval_episodes)
        print("Training episodes: {}, Avg rewards: {}".format(
            cur_episode, avg_rewards
        ))

    # Get the initial state of the game.
    time_step = env.reset()
    # As long as the game has not finished, do:
    while not time_step.last():
        # Each agent should choose an action and learn from the state it is in (time_step)
        agent_output = [agents[player_id].step(time_step, is_evaluation=True) for player_id in range(num_players)]
        # Do the chosen actions and get the new state.
        time_step = env.step([x.action for x in agent_output])

    # Episode is done
    # Let each player learn from the outcome of the episode.
    for agent in agents:
        agent.step(time_step)



## Run the game when the players have 'learned'.
# Get the initial state of the game.
time_step = env.reset() 
# As long as the game has not ended, do:
while not time_step.last():
    # Each agent should choose an action and learn from the state it is in (time_step)
    agent_output = [agents[player_id].step(time_step, is_evaluation=True) for player_id in range(num_players)]
    # Do the chosen actions and get the new state.
    time_step = env.step([x.action for x in agent_output])
print("The utilities for the players after their final game (after learning) are: {}".format(time_step.rewards))
