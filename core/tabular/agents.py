import numpy as np
import math

def value_iteration(P, r, discount, num_iterations, init_constant=0):
    """
    Value iteration
    """
    num_states, num_actions, _ = P.shape
    q = np.zeros((num_states, num_actions)) + init_constant
    for _ in range(num_iterations):
        q = r + discount * (P @ q.max(axis=-1, keepdims=True)).squeeze(-1)
    return q


def softmax(q, tau=0.1):
    #   return tau * np.log(np.exp(q / tau).sum(-1, keepdims=True))
    q_max = q.max(axis=-1, keepdims=True)
    return tau * np.log(np.exp((q - q_max) / tau).sum(-1, keepdims=True)) + q_max


def soft_value_iteration(P, r, tau, discount, num_iterations, init_constant=0):
    """
    Soft value iteration
    """
    num_states, num_actions, _ = P.shape
    q = np.zeros((num_states, num_actions)) + init_constant
    for _ in range(num_iterations):
        q = r + discount * (P @ softmax(q, tau=tau)).squeeze(-1)
    return q


def fqi(P, r, beta_as, discount, num_iterations, init_constant=0):
    mask = np.where(beta_as > 0)
    num_states, num_actions, _ = P.shape
    q = np.zeros((num_states, num_actions)) + init_constant
    for _ in range(num_iterations):
        q[mask] = (r + discount * (P @ q.max(axis=-1, keepdims=True)).squeeze(-1))[mask]
    return q, q.argmax(axis=-1)


def sarsa(P, r, beta_as, discount, num_iterations, init_constant=0):
    """
    Policy Evaluation
    """
    mask = np.where(beta_as > 0)
    num_states, num_actions, _ = P.shape
    q = np.zeros((num_states, num_actions)) + init_constant
    for _ in range(num_iterations):
        q[mask] = (r + discount * (P @ (beta_as * q).sum(axis=-1)))[mask]
    return q, q.argmax(axis=-1)


def in_sample(P, r, beta_as, tau, discount, num_iterations, init_constant=0):
    zero_mask = np.where(beta_as <= 0)
    beta_as[zero_mask] += 1e-8
    log_beta_as = np.log(beta_as)
    beta_as[zero_mask] = 0

    num_states, num_actions, _ = P.shape
    q = np.zeros((num_states, num_actions)) + init_constant
    for _ in range(num_iterations):
        q[zero_mask] = 0
        qmax = q.max(axis=-1, keepdims=True)
        v = tau * np.log(
            np.clip((beta_as * np.exp((q - qmax) / tau - log_beta_as)).sum(axis=-1, keepdims=True),
                    a_min=1e-08, a_max=np.inf)
        ) + qmax
        q = r + discount * (P @ v).squeeze(-1)

    qmax = q.max(axis=-1, keepdims=True)
    pi = beta_as * np.exp((q - qmax) / tau - log_beta_as)
    pi[zero_mask] += 1e-8
    pi /= pi.sum(axis=-1, keepdims=True)
    pi = np.array([np.random.choice(np.arange(num_actions), p=pi[i]) if pi[i].sum() > 0 else False for i in range(len(pi))])
    return q, pi


def in_sample_max(P, r, beta_as, tau, discount, num_iterations, init_constant=0):
    mask = np.where(beta_as > 0)
    num_states, num_actions, _ = P.shape
    q = np.zeros((num_states, num_actions)) + init_constant
    q[np.where(beta_as == 0)] = -1e8
    for _ in range(num_iterations):
        q[mask] = (r + discount * (P @ q.max(axis=-1, keepdims=True)).squeeze(-1))[mask]
    return q, q.argmax(axis=-1)