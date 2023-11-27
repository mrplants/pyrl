from abc import ABC, abstractmethod
from pyrl.agents.base import Agent

class DiscreteAgent(Agent):
    """
    Abstract base class for agents that operate in environments with discrete
    state and action spaces.
    """

    def __init__(self, num_states, num_actions):
        """
        Initialize the DiscreteSpaceAgent with the number of states and actions.

        Args:
            num_states (int): The number of states in the environment.
            num_actions (int): The number of actions in the environment.
        """
        self.num_states = num_states
        self.num_actions = num_actions

    @abstractmethod
    def step(self, state, reward):
        """
        Abstract method to be implemented by subclasses.
        Defines the behavior of the agent for a single step in the environment.

        Args:
            state (int): The current observed state.
            reward (float): The reward received from the previous action.

        Returns:
            int: The action to be taken.
        """
        pass

    @abstractmethod
    def update_policy(self):
        """
        Abstract method to be implemented by subclasses.
        Updates the policy based on learned values, if applicable.
        """
        pass
