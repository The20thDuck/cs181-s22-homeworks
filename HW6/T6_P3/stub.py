# Imports.
from http.client import NON_AUTHORITATIVE_INFORMATION
from urllib.parse import non_hierarchical
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt

# uncomment this for animation
from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey as SwingyMonkeyNoAnim

NO_ANIMATION = True
EPOCH_STEP = 10
SEP_GRAVITY = True

X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900

gravities = []

class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_vel = None
        self.gravity = 0
        
        self.iteration = 0

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q_unk = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))
        self.Q1 = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))
        self.Q4 = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    def reset(self):
        gravities.append(self.gravity)
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_vel = None
        self.gravity = 0
        
        self.iteration = 0


    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # TODO (currently monkey just jumps around randomly)
        # 1. Discretize 'state' to get your transformed 'current state' features.
        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        # 3. Choose the next action using an epsilon-greedy policy.

        alpha = self.alpha
        # update epsilon
        self.iteration += 1
        epsilon = self.epsilon/self.iteration
        gamma = self.gamma

        ## Check gravity
        if self.gravity == 0:
            vel = state["monkey"]["vel"]
            if self.last_vel is not None:
                if self.last_vel - vel in [1, 4]:
                    self.gravity = self.last_vel - vel
                    # print(self.gravity)
            self.last_vel = vel

        # if SEP_GRAVITY:

        if not SEP_GRAVITY or self.gravity == 0:
            Q = self.Q_unk
        elif self.gravity == 1:
            Q = self.Q1
        elif self.gravity == 4:
            Q = self.Q4
        else:
            print("gravity not 1 or 4?")

        x, y = self.discretize_state(state)

        # perform update
        if self.last_state != None:
            last_x, last_y = self.discretize_state(self.last_state)

            new_q = self.last_reward + gamma*np.max(Q[:, x, y])
            td = new_q - Q[self.last_action, last_x, last_y]

            Q[self.last_action, last_x, last_y] += alpha * td


        e = npr.rand()
        if e < epsilon:
            # pick random action
            new_action = npr.randint(2)
        else:
            # pick greedily
            new_action = np.argmax(Q[:, x, y])

        # new_action = npr.rand() < 0.1
        new_state = state

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        if (ii % EPOCH_STEP != 0 or NO_ANIMATION):
            swing = SwingyMonkeyNoAnim(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)
        else:
            swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    num_epochs = 100
    num_iters = 50

    # Find optimal alpha, gamma, epsilon
    '''
    alphas = [0.01, 0.1, 0.5]
    gammas = [0.1, 0.5, 0.9, 0.99]
    epsilons = [1, 0.3, 0.1, 0.01]

    histories = np.zeros((num_iters, num_epochs))
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                for i in range(num_iters):
                    agent = Learner(alpha, gamma, epsilon)
                    hist = []
                    run_games(agent, hist, num_epochs, 5)
                    histories[i, :] = hist
                print(alpha, gamma, epsilon, np.mean(np.sum(histories, axis=1)))
    
    '''

    # Empty list to save history.
    '''
    histories = np.zeros((2, num_iters, num_epochs))
    for j in range(2):
        SEP_GRAVITY = [True, False][j]

        for i in range(num_iters):
            agent = Learner(0.1, 0.99, 0.01)
            hist = []
            run_games(agent, hist, num_epochs, 5)
            histories[j, i, :] = hist
        print("Gravity:", SEP_GRAVITY, np.mean(np.sum(histories[j], axis=1)))
    plt.plot(np.arange(num_epochs), np.mean(histories[0], axis = 0), label="Inferred Gravity")
    plt.plot(np.arange(num_epochs), np.mean(histories[1], axis = 0), label="Regular")
    plt.legend(loc="upper left")
    plt.title("Avg Score vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.show()
    '''
    agent = Learner(0.01, 0.99, 0.01)

    # Run games. You can update t_len to be smaller to run it faster.
    SEP_GRAVITY = True
    hist = []
    run_games(agent, hist, 100, 50)
    print(hist)
    plt.plot(np.arange(num_epochs), hist)
    plt.legend(loc="upper left")
    plt.title("Score vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.show()
    # Save history. 
    # np.save('hist', np.array(hist2))
