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
    # Each value in the table is a dictionary 
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
    for stateString in symmetricalStates(state.dbn_string(),num_rows, num_cols):
        if not (stateString in transpTable.keys()):
            transpTable[stateString] = dict()
        k = transpTable[stateString] 
        k[len(score)] = result
    return result


def minimax_search(game,
                   state=None,
                   maximizing_player_id=None,
                   state_to_key=lambda state: state):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    [[ EXTENSION ]]
    * Transposition table
    * Symmetries

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

    # Initialise the transposition table as a dictionary.
    transpTable=dict()

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

    return v

def symmetricalStates(state, nrRows, nrCols):
    """
    Finds the symmetrical states of the given state.

    These symmetrical states include:
    - the state itself
    - the horizontally mirrored state
    - the vertically mirrored state
    - the vertically and horizontally mirrored state
    - the diagonally top left - bottom right mirrored state (for squared game boards)
    - the diagonally bottom left - top right mirrored state (for squared game boards)

    Arguments:
      state: The state to mirror.
      nrRows: the number of rows of the game board
      nrCols: the number of columns of the game board

    Returns:
      A set with the symmetrical states of the given state (including the given state).
    """

    # Preprocessing
    # Divide the given state in groups of lines, according to:
    #   - rows: each row is a row of horizontal lines on the game board
    #           -> size: nrRows + 1
    #           -> each row is a string of length nrCols 
    #   - columns: each column is a column of vertical lines on the game board
    #           -> size: nrRows
    #           -> each column is a string of length nrCols+1
    rows = []
    columns = []
    for i in range(nrRows+1):
        rows.append(state[i*(nrCols):(i+1)*(nrCols)])
    for j in range(nrRows):
        columns.append(state[(nrRows+1)*nrCols + j*(nrCols+1):(nrRows+1)*nrCols + (j+1)*(nrCols+1)])

    # The set of mirrored states, including the given state.
    res = set()
    res.add(state)


    # Vertically mirrored state
    rows1 = [row[::-1] for row in rows]
    columns1 = [column[::-1] for column in columns]
    state1 = ''.join(rows1)+''.join(columns1)
    res.add(state1)

    # Horizontally mirrored state
    rows2 = rows[::-1]
    columns2 = columns[::-1]
    state2 = ''.join(rows2)+''.join(columns2)
    res.add(state2)

    # Vertically and horizontally mirrored state
    rows3 = rows1[::-1]
    columns3 = columns1[::-1]
    state3 = ''.join(rows3)+''.join(columns3)
    res.add(state3)

    # Check if square board
    if (nrRows == nrCols):
        # Diagonally mirrored (top left - bottom right)
        rows4 = []
        for j in range(nrRows+1):
            rows4.append(''.join(column[j] for column in columns))
        columns4 = []
        for j in range(nrRows):
            columns4.append(''.join(row[j] for row in rows))
        state4 = ''.join(rows4)+''.join(columns4)
        res.add(state4)

        # Diagonally mirrored (bottom left - top right)
        rows5 = [row[::-1] for row in rows4][::-1]
        columns5 = [column[::-1] for column in columns4][::-1]
        state5 = ''.join(rows5)+''.join(columns5)
        res.add(state5)

        # Horizontally mirrored states of the diagonally mirrored states
        rows6 = rows4[::-1]
        columns6 = columns4[::-1]
        state6 = ''.join(rows6)+''.join(columns6)
        res.add(state6)
        rows7 = rows5[::-1]
        columns7 = columns5[::-1]
        state7 = ''.join(rows7)+''.join(columns7)
        res.add(state7)


    return res


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

        if value == 0:
            print("It's a draw")
        else:
            winning_player = 1 if value == 1 else 2
            print(f"Player {winning_player} wins.")
        
        res.append(end-start)
        print(end-start)

    # Take the average of the different execution times.
    print(f"Execution time: {sum(res)/len(res)}")


if __name__ == "__main__":
    app.run(main)
