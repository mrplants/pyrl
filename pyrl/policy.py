from .types import Context, Action, Reward
from .memory import Memory
from typing import Generic, NewType

class BasePolicy(Generic[Context, Action, Reward]):
    """ Abstract class for policies.
    """
    def sample_action(self, context:Context) -> Action:
        """ Samples an action from the policy.

        Args:
            context (Context): The environment context, based on observations.

        Returns:
            Action: The action to take in the environment.
        """
        raise NotImplementedError

    def update(self, memory:Memory) -> None:
        """ Updates the policy.

        Args:
            memory (Memory): The memory of the agent.
        """
        raise NotImplementedError

Policy = NewType('Policy', BasePolicy)