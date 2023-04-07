def printState(state, nrRows, nrCols):
    game_string = (f"dots_and_boxes(num_rows={nrRows},num_cols={nrCols},"
                "utility_margin=true)")
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state(state)
    print(state)


def printTable(transpTable,nrRows, nrColumns):
    for key in transpTable.keys():
        printState(key, nrRows, nrColumns)
        print(transpTable[key])

def part2num(part):
    p = {'h': 0, 'horizontal': 0,  # Who has set the horizontal line (top of cell)
         'v': 1, 'vertical':   1,  # Who has set the vertical line (left of cell)
         'c': 2, 'cell':       2}  # Who has won the cell
    return p.get(part, part)
def state2num(state):
    s = {'e':  0, 'empty':   0,
         'p1': 1, 'player1': 1,
         'p2': 2, 'player2': 2}
    return s.get(state, state)
def num2state(state):
    s = {0: 'empty', 1: 'player1', 2: 'player2'}
    return s.get(state, state)
def get_observation(obs_tensor, state, row, col, part, num_cols, num_rows):
    num_parts = 3
    num_cells = (num_rows + 1) * (num_cols + 1)
    state = state2num(state)
    part = part2num(part)
    idx =   part \
          + (row * (num_cols + 1) + col) * num_parts  \
          + state * (num_parts * num_cells)
    return obs_tensor[idx]
def get_observation_state(obs_tensor, row, col, part, num_cols, num_rows , as_str=True):
    is_state = None
    for state in range(3):
        if get_observation(obs_tensor, state, row, col, part, num_cols, num_rows) == 1.0:
            is_state = state
    if as_str:
        is_state = num2state(is_state)
    return is_state


def getScore(state, nrRows, nrCols):
    obsTensor = state.observation_tensor()
    score = {'empty':0, 'player1':0, 'player2':0}
    for i in range(nrRows):
        for j in range(nrCols):
            idx = get_observation_state(obsTensor, i,j, 2, nrCols, nrRows)
            score[idx] = score[idx] + 1
    return {0: score['player1'], 1: score['player2']}
