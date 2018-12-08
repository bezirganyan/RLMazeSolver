import numpy as np
import matplotlib.pyplot as plt
import time
# from IPython import display
import seaborn as sns
from envs.MazeGridWorld import MazeGridWorld

def calculate_action_expectation(state, action, Values):
    ps = env.next_states(state, action)
    summation = 0
    for s, p in ps:
        summation += Values[s] * p

    return summation

def calculate_V(V, gamma=0.99, iter_count=100, visual=False):
    for _ in range(iter_count):
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if env.field[i, j] != 1:
                    V[i, j] = max([R[i, j] + gamma*(calculate_action_expectation((i, j), a, V)) for a in range(4)])
                if visual:
                    plt.figure()
                    sns.heatmap(V.T)
                    display.clear_output(wait=True)
                    display.display(plt.gcf())
                    time.sleep(0.001)

    return V

def policy(state):
    choices = []
    for action in range(4):
        a = env.next_states(state, action)
        probs = []
        for i in a:
            probs.append(i[1])
        new_state = a[np.argmax(probs)][0]
        choices.append(V[new_state])

    return np.argmax(choices)
