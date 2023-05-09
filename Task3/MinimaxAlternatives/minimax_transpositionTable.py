from sys import getsizeof
import pyspiel
from absl import app
from auxilaryMethods import *
import time

def _minimax(state, maximizing_player_id, transpTable: dict, num_rows, num_cols, score = [], action = 0):
    """
    Implements a min-max algorithm

    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.
      transpTable: The current transposition table (dictionary) of the game.
      num_rows: the number of rows of the game board
      num_cols: the number of columns of the game board
      score: A list with the cells won by player 1
      action: The last chosen action

    Returns:
      The optimal value of the sub-game starting in state
    """

    if state.is_terminal():
        return state.player_return(maximizing_player_id)
    
    # The key of the table is the lines that are filled in.
    # Each value in the table is a dictionary 1
    # with the value of the score as key and the minimax value as value.
    key = state.dbn_string()
    # Update the list of scored cells after the last action
    score = updateScore(state,num_rows, num_cols, score, action)
    if (key in transpTable.keys()):
        k = transpTable[key]
        if (len(score) in k.keys()):
            # Current state and score already found
            return k[len(score)]
    else:
        # Current state not found,
        # so initialise a new dictionary for this state.
        transpTable[key] = dict()
        k = transpTable[key] 

    player = state.current_player()
    if player == maximizing_player_id:
        selection = max
    else:
        selection = min
    values_children = [_minimax(state.child(action), maximizing_player_id, transpTable, num_rows, num_cols, score, action) for action in state.legal_actions()]

    # Store the found value.
    result = selection(values_children)
    k[len(score)] = result
    return result


def minimax_search(game,
                   transpTable=dict(),
                   state=None,
                   maximizing_player_id=None,
                   state_to_key=lambda state: state):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    [[ EXTENSION ]]
    * Transposition table

    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.

    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """
    game_info = game.get_type()

    if game.num_players() != 2:
        raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("The game must be a Deterministic one, not {}".format(
            game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
        raise ValueError(
            "The game must be a perfect information one, not {}".format(
                game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("The game must be turn-based, not {}".format(
            game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("The game must be 0-sum, not {}".format(game.utility))

    if state is None:
        state = game.new_initial_state()
    if maximizing_player_id is None:
        maximizing_player_id = state.current_player()

    # Get the number of rows and columns of the game board.
    params = game.get_parameters()
    num_rows = params['num_rows']
    num_cols = params['num_cols']

    v = _minimax(
        state.clone(),
        maximizing_player_id=maximizing_player_id,
        transpTable=transpTable,
        num_rows= num_rows,
        num_cols=num_cols)
    
    # For debugging:
    # printTable(transpTable, 1,2)

    return v, transpTable

def main(_):
    # The number of times to measure the execution time (and averaging afterwards)
    n = 20

    # The number of rows and columns of the game board
    num_rows = 3
    num_cols = 3

    # A list with the measured execution times
    res = []
    size = []
    keys = []

    games_list = pyspiel.registered_names()
    assert "dots_and_boxes" in games_list
    game_string = f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols})"

    print("Creating game: {}".format(game_string))
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()
    print(state.dbn_string())

    # Initialise the transposition table as a dictionary.
    transpTable=dict()
    print(f"Len: {getsizeof(transpTable)}")

    for i in range(n):
        start = time.time()
        print("Playing game " + str(i))
        transpTable=dict()
        value, transpTable = minimax_search(game, transpTable)

        end = time.time()

        if value == 0:
            print("It's a DRAW")
        else:
            winning_player = 1 if value == 1 else 2
            print(f"Player {winning_player} WINS.")
        
        print(f"Size: {getsizeof(transpTable)}")
        print(f"Len: {len(transpTable)}")
        print("Exe time: " + str(end-start))

        res.append(end-start)
        size.append(getsizeof(transpTable))
        keys.append(len(transpTable))

    # Take the average of the different execution times.
    print(f"Average execution time: {round(sum(res)/len(res),10)}")
    # Take the average of the different transp table sizes in bytes.
    print(f"Average dict size: {sum(size)/len(size)}")
    # Take the average of the different nr of keys.
    print(f"Avg nr of keys {sum(keys)/len(keys)}")

if __name__ == "__main__":
    app.run(main)
