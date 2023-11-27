from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for different types of agents.
    """

    @abstractmethod
    def step(self, observation, reward):
        """
        Abstract method to be implemented by subclasses. 
        Defines the behavior of the agent for a single step in the environment.

        Args:
            observation (int): The current observed state.
            reward (float): The reward received from the previous action.

        Returns:
            int: The action to be taken.
        """
        pass