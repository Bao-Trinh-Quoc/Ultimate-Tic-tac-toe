import numpy as np
from collections import defaultdict
import copy
from state import UltimateTTT_Move
from state import State, State_2

class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        """
        """
    def untried_actions(self):
        """
        """
    def win_rate(self):
        """
        """
    def visited_times(self):
        """
        """
    def expand(self):
        """
        """
    def is_terminal_node(self):
        """
        """
    def rollout(self):
        """
        """
    def backpropagate(self, result):
        """
        """
    def is_fully_expanded(self):
        """
        """
    def best_child(self, c_param=0.1):
        """
        """
    def rollout_policy(self, possible_moves):
        """
        """
    def tree_policy(self):
        """
        """
    def best_action(self):
        """
        """
    