import unittest
import numpy as np
from pyrl.util import value_iteration

class TestValueIteration(unittest.TestCase):

    def test_trivial_mdp(self):
        # Define a trivial MDP environment
        P = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # Transition probabilities
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        R = np.array([0, 1, 0])  # Rewards for each state-action pair
        gamma = 0.99  # Discount factor
        threshold = 0.01  # Convergence threshold

        # Expected values calculated manually for the trivial MDP
        expected_V = np.array([1.0, 0.0, 0.0])
        expected_Q = np.array([[0.99, 1.0], 
                                [0.0, 0.0], 
                                [0.0, 0.0]])

        # Run value iteration
        V, Q = value_iteration(P, R, gamma, threshold)

        # Check if the computed V and Q are close to the expected values
        np.testing.assert_array_almost_equal(V, expected_V, decimal=2)
        np.testing.assert_array_almost_equal(Q, expected_Q, decimal=2)

# The value_iteration function would be imported from the module where it's defined.

if __name__ == '__main__':
    unittest.main()
