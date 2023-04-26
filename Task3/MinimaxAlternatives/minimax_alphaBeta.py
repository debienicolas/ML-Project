import pyspiel
from open_spiel.python.algorithms.minimax import alpha_beta_search
import time
from absl import app

def minimax_search(game,
                   state=None,
                   maximizing_player_id=None,
                   state_to_key=lambda state: state):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    [[ EXTENSION ]]
    * alpha-beta pruning

    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.

    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """
    return alpha_beta_search(game, state, maximizing_player_id= maximizing_player_id)

def main(_):
    # The number of times to measure the execution time (and averaging afterwards)
    n = 20
    
    # The number of rows and columns of the game board
    num_rows = 2
    num_cols = 3

    # A list with the measured execution times
    res = []

    games_list = pyspiel.registered_names()
    assert "dots_and_boxes" in games_list
    game_string = f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols})"

    print("Creating game: {}".format(game_string))
    game = pyspiel.load_game(game_string)

    for i in range(n):
        start = time.time()

        value = minimax_search(game)

        end = time.time()

        if value[0] == 0:
            print("It's a draw")
        else:
            winning_player = 1 if value[0] == 1 else 2
            print(f"Player {winning_player} wins.")
        
        res.append(end-start)
        print(end-start)

    # Take the average of the different execution times.
    print(f"Execution time: {sum(res)/len(res)}")

if __name__ == "__main__":
    app.run(main)
