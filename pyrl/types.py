from typing import TypeVar
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

# State fully represents an environment at a specific point in time
State = TypeVar('State')
# Context represents the agent's prediction of the environment state
# It represents all historical transitions and enables the Markov assumption
Context = TypeVar('Context')
# Observation is a partial representation of the environment
# It contains only the information that the agent can observe
Observation = TypeVar('Observation')
# Action represents the action that the agent can take in the environment
Action = TypeVar('Action')
# Reward represents the reward that the agent receives upon entering a state
Reward = TypeVar('Reward')