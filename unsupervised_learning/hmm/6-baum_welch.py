#!/usr/bin/env python3
"""Create the function that performs the Baum-Welch algorithm for a
Hidden Markov Model"""

import numpy as np

def forward(Observations, Transition, Emission, Initial):
    T = Observations.shape[0]
    N = Transition.shape[0]
    
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observations[0]]
    
    for t in range(1, T):
        for n in range(N):
            F[n, t] = np.dot(F[:, t - 1], Transition[:, n]) * Emission[n, Observations[t]]
    
    return F

def backward(Observations, Transition, Emission):
    T = Observations.shape[0]
    N = Transition.shape[0]
    
    B = np.zeros((N, T))
    B[:, T - 1] = 1
    
    for t in range(T - 2, -1, -1):
        for n in range(N):
            B[n, t] = np.sum(B[:, t + 1] * Transition[n, :] * Emission[:, Observations[t + 1]])
    
    return B

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
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    T = Observations.shape[0]
    
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    M, M_check = Transition.shape
    if M != M_check:
        return None, None
    
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    M_check, N = Emission.shape
    if M_check != M:
        return None, None
    
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2 or Initial.shape[1] != 1:
        return None, None
    M_check, one = Initial.shape
    if M_check != M or one != 1:
        return None, None
    
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    
    for _ in range(iterations):
        # Forward pass
        alpha = forward(Observations, Transition, Emission, Initial)
        
        # Backward pass
        beta = backward(Observations, Transition, Emission)
        
        # Calculate xi and gamma
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, t].T, Transition) * Emission[:, Observations[t + 1]].T, beta[:, t + 1])
            for i in range(M):
                numerator = alpha[i, t] * Transition[i, :] * Emission[:, Observations[t + 1]].T * beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        P_T = np.sum(alpha[:, T - 1])
        gamma = np.hstack((gamma, alpha[:, T - 1][np.newaxis].T / P_T))
        
        # Update Transition and Emission matrices
        Transition = np.sum(xi, 2) / np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))
        denominator = np.sum(gamma, axis=1)
        for l in range(N):
            Emission[:, l] = np.sum(gamma[:, Observations == l], axis=1)
        Emission = np.divide(Emission, denominator.reshape((-1, 1)))

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
