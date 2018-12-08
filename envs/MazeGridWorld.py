import numpy as np
import random
import matplotlib.pyplot as plt
import time
from IPython import display

class MazeGridWorld:
    def __init__(
            self,
            grid_size=(30,30),
            stochasticity=0.1,
            visual=False):
        self.w, self.h = grid_size
        self.stochasticity = stochasticity
        self.visual = visual

        self.generate_maze(grid_size[0], grid_size[1])

        self.reset()

    def generate_maze(self, w, h):

        def getBlock(x,y, m):
            if x>=0 and y>=0 and x<w and y<h:
                return m[x, y]
            else:
                return 0

        def setBlock(x,y,b, m):
            if x>=0 and y>=0 and x<w and y<h:
                m[x, y] = b

        def makePassage(x, y, i, j, m):
            if getBlock(x+i, y+j, m) == 1 and\
                getBlock(x+i+j, y+i+j, m) == 1 and\
                getBlock(x+i-j, y+j-i, m) == 1:
                    if getBlock(x+i+i, y+j+j, m) == 1 and\
                        getBlock(x+i+i+j, y+i+j+j, m) == 1 and\
                        getBlock(x+i+i-j, y+j+j-i, m) == 1:
                            if random.random() > 0.2:
                                setBlock(x+i, y+j, 0, m)

            return m

        maze = np.zeros((w, h))

        for i in range(w):
            maze[i, :] = 1

        maze[0, 1] = 0;
        entery = (0, 1)
        goal = None

        for _ in range (100):
            for y in range(h):
                for x in range(w):
                    if maze[x, y] == 0:
                        maze = makePassage(x,y, -1,0, maze)
                        makePassage(x,y,  1,0, maze)
                        makePassage(x,y, 0,-1, maze)
                        makePassage(x,y, 0, 1, maze);

        while True:
            empty = random.randint(1, w-1)
            if maze[empty, h-2] == 0:
                maze[empty, h-1] = 3;
                goal = (empty, h-1)
                break

        self.maze = maze
        self.goal = goal
        self.entery = entery

    def reset(self):
        """ resets the environment
        """
        self.field = self.maze
        self.field[self.entery] = 2
        self.field[self.goal] = 3
        self.pos = self.entery
        state = self.get_state()
        return state

    def step(self, a):
        """ take a step in the environment
        """

        if np.random.rand() < self.stochasticity:
            a = np.random.randint(4)

        self.field[self.pos] = 0
        self.pos = self.move(a)
        self.field[self.pos] = 2

        done = False
        reward = 0
        if self.pos == self.goal:
            # episode finished successfully
            done = True
            reward = 1
        next_state = self.get_state()
        return next_state, reward, done

    def clip_xy(self, x, y):
        """ clip coordinates if they go beyond the grid
        """
        x_ = np.clip(x, 0, self.w - 1)
        y_ = np.clip(y, 0, self.h - 1)
        return x_, y_

    def isValidCoordinate(self, x, y):
        if x >= 0 and \
            y >= 0 and \
            x < self.w and \
            y < self.h and \
            self.field[x, y] != 1:
                return True
        else:
            return False

    def move(self, a):
        """ find valid coordinates of the agent after executing action
        """
        x, y = self.pos
        self.field[x, y] = 0
        if a == 0:
            x_, y_ = x + 1, y
        if a == 1:
            x_, y_ = x, y + 1
        if a == 2:
            x_, y_ = x - 1, y
        if a == 3:
            x_, y_ = x, y - 1
        # check if new position does not conflict with the wall
        if not self.isValidCoordinate(x_, y_):
            x_, y_ = x, y
        return self.clip_xy(x_, y_)

    def get_state(self):
        """ get state of the environment
        """
        if self.visual:
            state = np.rot90(self.field)[:, :, None]
        else:
            state = self.pos
        return state

    def draw_state(self):
        """ draws grid world
        """
        img = np.rot90(1-self.field)
        plt.imshow(img, cmap="gray")

    def next_states(self, state, action):
        """
        Parameters
        ----------
        state (s): environment state
        action (a): action taken in state

        Returns
        -------
        list of pairs [s', p(s'|s,a)]
        s': possible next state
        p(s'|s,a): transition probability
        """
        x, y = self.pos

        self.pos = state
        next_states = []
        for a in range(4):
            x_, y_ = self.move(a)
            prob = self.stochasticity / 4
            if a == action:
                prob += (1 - self.stochasticity)
            next_states.append([(x_, y_), prob])
        self.pos = (x, y)
        return next_states

    def play_with_policy(self, policy, max_iter=100, visualize=True):
        """ play with given policy
        Parameters
        ----------
        policy: function: state --> action
        max_iter: maximum number of time steps
        visualize: bool, if True visualize episode
        """
        self.reset()
        for i in range(max_iter):
            state = self.get_state()
            action = policy(state)
            next_state, reward, done = self.step(action)

            # plot grid world state
            if visualize:
                self.draw_state()
                display.clear_output(wait=True)
                display.display(plt.gcf())
                time.sleep(0.01)
            if done:
                break
        if visualize:
            display.clear_output(wait=True)
        return reward
