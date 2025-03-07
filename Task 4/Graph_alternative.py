import torch
import torch_geometric.data as geom_data
import numpy as np
import pyspiel



def state_to_graph_data(state):
    """
    This function takes a state and returns a PyTorch Geometric Data object
    in the perspective of the current player of the state (so the first player)
    """
    game = state.get_game()
    cols, rows = game.get_parameters()["num_cols"], game.get_parameters()["num_rows"]
    num_nodes = cols*rows
    # adding dummy node
    num_nodes += 1

    # Node features (x): node ownership and valency
    x = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j
            # Initialize the owner of the node
            owner = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j, part='c')
            # Put in the first player perspective
            if state.current_player() == 1 & owner != 0:
                owner = 3 - owner
            
            # if current player is 0 => if owner is 0 then it is 0, else if it is 1 it is 1, if it is 2 then it is -1
            if owner == 2:
                owner = -1

            x[node_index, 0] = owner
            # Node valency 
            x[node_index, 1] = 4 - getFilledLines(game,state.observation_tensor(),i,j)

    # Edge indices
    # Coins and string representation
    # for edges that have strings to no other node, we add a edge to the node itself
    # try to follow the same order as the actions in openspiel, first all the horizontal edges, then all the vertical edges
    edge_index = []
    edge_attr = []
    # edge attributes -> state of the graph

   # Top and bottom strings first (vertical strings)
    dummy_node_index = num_nodes - 1
    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j
            # if first row than you have to add the top string to the same node
            if i == 0:
                #edge_index.append([node_index, node_index])
                edge_index.append([node_index, dummy_node_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j, part='h')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                if edge_state == 2:
                    edge_state = 1
                edge_attr.append(edge_state)
            
            # from first row till the last row you have to add the bottom string with another node
            if i < rows - 1:
                bottom_neighbor_index = (i+1)*(cols) + j
                edge_index.append([node_index, bottom_neighbor_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i+1, col=j, part='h')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                if edge_state == 2:
                    edge_state = 1
                edge_attr.append(edge_state)

            # last row has to set the bottom string to the same node
            if i == rows - 1:
                #edge_index.append([node_index, node_index])
                edge_index.append([node_index, dummy_node_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i+1, col=j, part='h')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                if edge_state == 2:
                    edge_state = 1
                edge_attr.append(edge_state)

    # Left and right strings (horizontal strings)

    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j

            if j == 0:
                # Left string
                #edge_index.append([node_index, node_index])
                edge_index.append([node_index, dummy_node_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j, part='v')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                if edge_state == 2:
                    edge_state = 1
                edge_attr.append(edge_state)

            if j < cols - 1:
                # Horizontal edge
                right_neighbor_index = i*(cols) + (j+1)
                edge_index.append([node_index, right_neighbor_index])
                # set the edge attribute if the edge is filled
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j+1, part='v')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                if edge_state == 2:
                    edge_state = 1
                edge_attr.append(edge_state)
            
            if j ==  cols - 1:                
                #edge_index.append([node_index, node_index])
                edge_index.append([node_index, dummy_node_index])
                # set the edge attribute if the edge is filled
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j+1, part='v')
                
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                if edge_state == 2:
                    edge_state = 1
                edge_attr.append(edge_state)


    # turn edge indices into a tensor
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
    # turn edge attribute into a tensor
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    edge_attr = edge_attr.view(-1, 1)
    # Batch information
    batch = torch.zeros(num_nodes, dtype=torch.int64)

    return geom_data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr,batch=batch)


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


def get_observation(game,obs_tensor, state, row, col, part): 
    num_rows, num_cols = game.get_parameters()["num_rows"], game.get_parameters()["num_cols"]
    num_cells = (num_rows + 1) * (num_cols + 1)
    num_parts = 3   # (horizontal, vertical, cell)
    num_states = 3  # (empty, player1, player2)
    state = state2num(state)
    part = part2num(part)
    idx =   part \
          + (row * (num_cols + 1) + col) * num_parts  \
          + state * (num_parts * num_cells)
    return obs_tensor[idx]

def get_observation_state(game,obs_tensor, row, col, part):
    is_state = None
    for state in range(3):
        if get_observation(game, obs_tensor, state, row, col, part) == 1.0:
            is_state = state
    return is_state

# get the amount of connections for a node that have been played
def getFilledLines(game,obs_tensor,row,col):
    connections = 4
    # check if the node is connected to the top
    connections -= get_observation(game,obs_tensor,state2num('empty'),row,col,'h')
    # check if the node is connected to the left
    connections -= get_observation(game,obs_tensor,state2num('empty'),row,col,'v')
    # check if the node is connected to the right
    connections -= get_observation(game,obs_tensor,state2num('empty'),row,col+1,'v')
    # check if the node is connected to the bottom
    connections -= get_observation(game,obs_tensor,state2num('empty'),row+1,col,'h')
    return connections



# the first two rows of the edges doesn't match with the actions
def edges_to_actions(state,edges):
    game = state.get_game()
    num_rows = game.get_parameters()["num_rows"]
    actions = []
    first_row = []
    second_row = []
    for i in range(num_rows*2):
        if i % 2 == 0:
            first_row.append(edges[i])
        else:
            second_row.append(edges[i])
    actions += first_row
    actions += second_row
    actions += edges[num_rows*2:]
    return actions

def actions_to_edges(state,actions):
    game = state.get_game()
    num_rows = game.get_parameters()["num_rows"]
    edges = []
    first_row = []
    second_row = []
    for i in range(2*num_rows):
        if i % 2 == 0:
            first_row.append(actions[i])
        else:
            second_row.append(actions[i])
    edges += first_row
    edges += second_row
    edges += actions[2*num_rows:]
    return edges