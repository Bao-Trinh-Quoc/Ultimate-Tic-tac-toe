import numpy as np
from collections import defaultdict
import copy
from state import UltimateTTT_Move
from state import State, State_2

"""
A class representing a node in the Monte Carlo Tree Search (MCTS) algorithm.
A MTCS is a tree search algorithm that is used in decision processes with a large search space.
The algorithm is based on the principle of random sampling of the search space.
There are four main phases in the MCTS algorithm:
    1. Selection: Start from root R and select successive child nodes until a leaf node L is reached.
    2. Expansion: Unless L ends the game with a win/loss for either player, create one (or more) child nodes and choose node C from one of them.
    3. Simulation: Play a random playout from node C.
    4. Backpropagation: Use the result of the playout to update information in the nodes on the path from C to R.
"""
class MonteCarloTreeSearchNode():
    """
    Initialize the MCTS node
    @param: state: the state of the game
    @param: parent: the parent node
    @param: parent_action: the action that led to the parent node
    @param children: list of the children nodes
    @param _number_of_visits: number of times the node has been visited
    @param _results: the results of the node (dict)
    @param _results[1]: the number of wins
    @param _results[-1]: the number of losses
    @param _untried_actions: the untried actions
    @return: None
    """
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return
    """
    Return the number of untries actions
    @param: None
    @return: number of untried actions
    """
    def untried_actions(self):
        self._untried_actions = self.state.get_valid_moves
        return self._untried_actions
    """
    Return the win rate of the node
    @param: None
    @return: wins - losses
    """
    def win_rate(self):
        wins = self._results[1]
        losses = self._results[-1]
        return wins - losses
    """
    Return the number of times the node has been visited
    @param: None
    @return: number of visits
    """
    def visited_times(self):
        return self._number_of_visits
    """
    Return True if the expansion is fully expanded, False otherwise
    @param: None
    @return: True if the node is fully expanded, False otherwise
    """
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    """
    Expansion phase of the MCTS algorithm
    @param: None
    @return 
    """
    def expand(self):
        action = self._untried_actions.pop()
        next_state = my_act_moves(self.state, action)
        child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node
    """
    is the node a terminal node i.e leaf node i.e the game is over ?
    @param: None
    @return: True if the node is a terminal node, False otherwise
    """
    def is_terminal_node(self):
        return self.state.game_over
    
    """
    Policy for simulation phase of the MCTS algorithm
    @param: possible_moves: list of possible moves
    @return: random move from the list of possible moves
    @note: Could be improved by using a better policy for more efficient search
    """   
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
    """
    Simulation phase of the MCTS algorithm
    @param: None
    @return: the result of the simulation
    """
    def rollout(self):
        current_rollout_state = copy.deepcopy(self.state)
        while not current_rollout_state.game_over:
            possible_moves = current_rollout_state.get_valid_moves
            if len(possible_moves) == 0:
                break
            action = self.rollout_policy(possible_moves)
            current_rollout_state = my_act_moves(current_rollout_state, action)
        return current_rollout_state.game_result(current_rollout_state.global_cells.reshape(3,3))
    """
    Backpropagation phase of the MCTS algorithm
    @param: result: the result of the simulation
    @return: None
    """
    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)
    """
    Selection phase of the MCTS algorithm
    Choose the best child node based on the UCT (Upper Confidence Bound) formula
    @param: c_param: the exploration parameter
    @return: the best child node
    """
    def best_child(self, c_param=0.1):
        if len(self.children) == 0:
            return self
        # UCT = win_rate / visited_times + c_param * sqrt(log(parent.visited_times) / visited_times)
        # score_weights = [((child.win_rate()/child.visited_times()) + c_param * np.sqrt((2 * np.log(self.visited_times()) / child.visited_times()))) for child in self.children] 
        score_weights = []
        for child in self.children:
            exploitation = child.win_rate() / child.visited_times()
            exploration = c_param * np.sqrt(2 * np.log(self.visited_times()) / child.visited_times())
            score = exploitation + exploration
            score_weights.append(score)
        
        return self.children[np.argmax(score_weights)]
    """
    Navigate the tree to find the best action
    @param: None
    @return: the best action
    """
    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                if len(current_node.children) == 0:
                    break 
                current_node = current_node.best_child()
        return current_node
    """
    Perform the MCTS algorithm
    """
    def best_action(self):
        no_simulations = 200
        for _ in range(no_simulations):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child(c_param=np.sqrt(2))

"""
My implementation for act_move function in state.py. 
Changed the function to return the new state after the move is made.
(Cannot changed in state.py due to the requirements of the project)
"""
def my_act_moves(state: State_2, move: UltimateTTT_Move):
    state = copy.deepcopy(state)
    if not state.is_valid_move(move):
        raise ValueError(
            "move {0} on local board is not valid".format(move)
        )
    local_board = state.blocks[move.index_local_board]
    local_board[move.x, move.y] = move.value

    state.player_to_move *= -1 
    state.previous_move = move

    if state.global_cells[move.index_local_board] == 0: # not 'X' or 'O'
        if state.game_result(local_board):
            state.global_cells[move.index_local_board] = move.value

    return state