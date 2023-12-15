from .memory import Memory
from .policy import Policy
from .types import Observation, Action, Reward
from typing import Generic

class Agent(Generic[Observation, Action, Reward]):
    """ Convenience class for different RL agents.

    Attributes:
        memory (Memory): The memory of the agent.
        policy (Policy): The policy of the agent.
    """

    def __init__(self, memory:Memory, policy:Policy) -> None:
        self.memory = memory
        self.policy = policy

    def step(self, observation:Observation) -> Action:
        """ Defines the behavior of the agent for a single step in the environment.

        Args:
            observation (Observation): The observation from the environment.

        Returns:
            Action: The action to take in the environment.
        """
        context = self.memory.get_context(observation)
        return self.policy.sample_action(context)

    def observe(self, observation:Observation, action:Action, next_observation:Observation, reward:Reward) -> None:
        """ Defines the behavior of the agent when observing a transition in the environment.

        Args:
            observation (Observation): The observation from the environment.
            action (Action): The action taken in the environment.
            next_observation (Observation): The observation after taking the action.
            reward (Reward): The reward received from the environment.
        """
        self.memory.add_transition(observation, action, next_observation, reward)
    
    def update(self) -> None:
        """ Updates the policy of the agent.
        """
        self.memory.update()
        self.policy.update(self.memory)