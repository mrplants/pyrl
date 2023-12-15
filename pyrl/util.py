import numpy as np

def value_iteration(P: np.ndarray,
                    R: np.ndarray,
                    gamma: float=0.999,
                    threshold: float=1e-5) -> (np.ndarray, np.ndarray):
    """ Performs value iteration to calculate and return the optimal policy values.

    Args:
        P (np.ndarray, shape=(S,A,S)): Transition probabilities.
        R (np.ndarray, shape=(S)): Rewards for each state.
        gamma (float): Discount factor.
        threshold (float): Convergence threshold.

    Returns:
        V, Q (np.ndarray, shape=(S)), (np.ndarray, shape=(S,A)): The optimal policy values and the optimal policy.
    """
    V = np.zeros(P.shape[0])
    print(V.shape)
    while True:
        Q = np.sum(P * (R + gamma * V)[np.newaxis,np.newaxis], axis=2)
        V_new = np.amax(Q, axis=1)
        if np.amax(np.abs(V_new-V)) < threshold:
            break
        V = V_new
    return V, Q
