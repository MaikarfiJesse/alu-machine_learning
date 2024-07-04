#!/usr/bin/env python3
"""Create the function that performs the Baum-Welch algorithm for a
Hidden Markov Model"""

import numpy as np

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a Hidden Markov Model

    parameters:
        Observations [numpy.ndarray of shape (T,)]:
            contains the index of the observations
            T: number of observations
        Transition [2D numpy.ndarray of shape (M, M)]:
            contains the initialized transition probabilities
            M: the number of hidden states
        Emission [numpy.ndarray of shape (M, N)]:
            contains the initialized emission probabilities
            N: number of output states
        Initial [numpy.ndarray of shape (M, 1)]:
            contains the initialized starting probabilities
        iterations [positive int]:
            the number of times expectation-maximization should be performed

    returns:
        the converged Transition, Emission
        or None, None on failure
    """
    # Check that Observations is the correct type and dimension
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    T = Observations.shape[0]
    
    # Check that Transition is the correct type and dimension
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    M, M_check = Transition.shape
    if M != M_check:
        return None, None
    
    # Check that Emission is the correct type and dimension
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    M_check, N = Emission.shape
    if M_check != M:
        return None, None
    
    # Check that Initial is the correct type and dimension
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2 or Initial.shape[1] != 1:
        return None, None
    M_check, one = Initial.shape
    if M_check != M or one != 1:
        return None, None
    
    # Check that iterations is a positive integer
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    
    # Placeholder implementation for Baum-Welch algorithm
    for i in range(iterations):
        # Perform one iteration of the Baum-Welch algorithm here
        pass
    
    # Return placeholder values for Transition and Emission
    return Transition, Emission

# Example usage
if __name__ == "__main__":
    Observations = np.array([0, 1, 0, 2, 3])
    Transition = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    Emission = np.array([
        [0.5, 0.4, 0.1],
        [0.1, 0.3, 0.6],
        [0.0, 0.1, 0.9]
    ])
    Initial = np.array([0.6, 0.3, 0.1]).reshape(-1, 1)
    T, E = baum_welch(Observations, Transition, Emission, Initial)
    if T is not None and E is not None:
        print(np.round(T, 2))
        print(np.round(E, 2))
