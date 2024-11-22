import numpy as np
from mcts import MonteCarloTreeSearchNode

# def select_move(cur_state, remain_time):
#     valid_moves = cur_state.get_valid_moves
#     if len(valid_moves) != 0:
#         return np.random.choice(valid_moves)
#     return None

def select_move(cur_state, remain_time):
    root = MonteCarloTreeSearchNode(state = cur_state)
    best_node = root.best_action()
    return best_node.parent_action