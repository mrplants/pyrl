from .policy import BasePolicy
from .memory import Memory
from .util import value_iteration
from .types import Transition
from typing import NewType
import numpy as np
import random

DiscreteContext = NewType('DiscreteContext', int)
DiscreteAction = NewType('DiscreteAction', int)
FloatReward = NewType('FloatReward', float)

class DiscretePolicy(BasePolicy[DiscreteContext, DiscreteAction, FloatReward]):
    """ A policy that uses discrete contexts and actions.

    Attributes:
        num_contexts (int): The number of contexts possible.
        num_actions (int): The number of actions possible.
        Q (np.ndarray, shape=(num_contexts, num_actions)): The Q-table.
    """
    
    def __init__(self, num_contexts:int, num_actions:int) -> None:
        """ Initializes the policy.

        Args:
            num_contexts (int): The number of contexts possible.
            num_actions (int): The number of actions possible.
        """
        self.num_contexts = num_contexts
        self.num_actions = num_actions
        self.Q = np.zeros((num_contexts, num_actions))
    
    def sample_action(self, context: DiscreteContext) -> DiscreteAction:
        """ Samples an action from the policy where multiple maximum values result in a random choice.
        
        Args:
            context (DiscreteContext): The environment context, based on observations.
            
        Returns:
            DiscreteAction: The action to take in the environment.
        """
        # Get the array of Q-values for the given context
        q_values = self.Q[context]
        
        # Find the indices of the maximum Q-value(s)
        max_indices = np.flatnonzero(q_values == q_values.max())
        
        # Randomly choose from the indices with maximum Q-value
        chosen_index = random.choice(max_indices)
        
        return DiscreteAction(chosen_index)

    def update(self, memory: Memory) -> None:
        """ Updates the policy using value iteration.

        Args:
            memory (Memory): The memory of the agent.
        """
        # Get the transition probabilities and rewards
        transitions = list(memory.iterate_context_transitions())
        P = self.get_transition_probabilities(transitions)
        R = self.get_rewards(transitions)
        
        # Perform value iteration
        _, Q = value_iteration(P, R)
        
        # Update the Q-table
        self.Q = Q
    
    def get_rewards(self, transitions: list[Transition]) -> np.ndarray:
        """ Gets the rewards from the transitions.

        Args:
            transitions (list): The list of transitions.

        Returns:
            np.ndarray, shape=(num_contexts,): The rewards.
        """
        # Initialize the rewards
        R = np.zeros((self.num_contexts,))
        context_counts = np.zeros((self.num_contexts,))
        
        # Iterate over the transitions
        for transition in transitions:
            # Get the context and action
            context = transition.state
            action = transition.action
            
            # Increment the reward
            R[context] += transition.reward
            context_counts[context] += 1
        
        # Normalize the rewards, making sure not to divide by zero
        R /= np.maximum(context_counts, 1)
        
        return R
    
    def get_transition_probabilities(self, transitions: list[Transition]) -> np.ndarray:
        """ Gets the transition probabilities from the transitions.

        Args:
            transitions (list): The list of transitions.

        Returns:
            np.ndarray, shape=(num_contexts, num_actions, num_contexts): The transition probabilities.
        """
        # Initialize the transition probabilities
        P = np.zeros((self.num_contexts, self.num_actions, self.num_contexts))
        
        # Iterate over the transitions
        for transition in transitions:
            # Get the context, action, and next context
            context = transition.state
            action = transition.action
            next_context = transition.next_state
            
            # Increment the transition probability
            P[context, action, next_context] += 1
        
        # Normalize the transition probabilities
        P /= np.sum(P, axis=2, keepdims=True)
        
        return P