from .types import Observation, Action, Reward, Transition, Context
from typing import Generic, NewType, Generator
from collections import deque
import numpy as np

class Memory(Generic[Observation, Context, Action, Reward]):
    """ Records Markov transitions.

    The simplest form of memory simply passes through observations as context.

    Attributes:
        transitions (list): The list of transitions.
    """
    def __init__(self) -> None:
        self._transitions = deque()

    def add_transition(self, observation:Observation, action:Action, next_observation:Observation, reward:Reward) -> None:
        """ Adds a transition to the memory.

        Args:
            observation (Observation): The observation from the environment.
            action (Action): The action taken in the environment.
            next_observation (Observation): The observation after taking the action.
            reward (Reward): The reward received from the environment.
        """
        self._transitions.append(Transition(observation, action, next_observation, reward))

    def get_context(self, observation:Observation) -> Context:
        """ Gets the context from the observation.

        Args:
            observation (Observation): The observation from the environment.

        Returns:
            Context: The context for the observation.
        """
        return observation
    
    def iterate_context_transitions(self) -> Generator[(Context, Transition), None, None]:
        """ Iterates over the context transitions.

        Yields:
            (Context, Transition): The context and transition.
        """
        for transition in self._transitions:
            yield Transition(self.get_context(transition.state),
                             transition.action,
                             self.get_context(transition.next_state),
                             transition.reward)
    
    def update(self) -> None:
        """ Updates the memory. In this case, does nothing."""
        pass