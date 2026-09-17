"""
snake_env.py

A lightweight, dependency-free Snake game environment built to follow the
same (reset, step, render) contract used by Gymnasium environments, so the
rest of the code (agent, training loop) is written the way it would be for
any standard RL environment.

State representation
---------------------
Instead of feeding the agent the raw pixel grid (which would require a much
bigger convolutional network and a lot more training time), the environment
exposes an 11-feature vector that is a compact, hand-engineered summary of
the game state:

    [0] danger straight ahead
    [1] danger to the right (relative to current heading)
    [2] danger to the left  (relative to current heading)
    [3] moving left
    [4] moving right
    [5] moving up
    [6] moving down
    [7] food is to the left of the head
    [8] food is to the right of the head
    [9] food is above the head
    [10] food is below the head

This is the same style of feature engineering commonly used in small
Snake-RL projects (see README / report references) because it keeps the
state space small enough for a compact MLP + tabular-style Q-learning /
DQN to learn a good policy in a few thousand episodes, while still giving
the agent everything it needs to make decisions.

Action space
------------
Actions are *relative* to the snake's current heading, not absolute
directions. This avoids the agent ever being able to "reverse into itself"
by construction, which both speeds up learning and matches how the game is
actually played:

    0 -> go straight
    1 -> turn right
    2 -> turn left
"""

import numpy as np
from collections import deque

# Grid directions as (dx, dy) with y growing downward (row, col style)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Clockwise ordering used to compute relative turns
CLOCKWISE = [UP, RIGHT, DOWN, LEFT]


class SnakeEnv:
    """A minimal, fast, NumPy-based Snake environment."""

    def __init__(self, grid_size: int = 12, max_steps_without_food: int = 100):
        self.grid_size = grid_size
        self.max_steps_without_food = max_steps_without_food
        self.action_space_n = 3
        self.observation_space_n = 11
        self.reset()

    # ------------------------------------------------------------------ #
    # Core Gym-style API
    # ------------------------------------------------------------------ #
    def reset(self):
        mid = self.grid_size // 2
        self.direction = RIGHT
        self.snake = deque([(mid, mid), (mid - 1, mid), (mid - 2, mid)])
        self.score = 0
        self.steps_since_food = 0
        self.total_steps = 0
        self._place_food()
        return self._get_state()

    def step(self, action: int):
        """
        action: 0 = straight, 1 = right turn, 2 = left turn
        returns: (next_state, reward, done, info)
        """
        self._turn(action)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        self.total_steps += 1
        self.steps_since_food += 1

        reward = -0.01  # small per-step penalty encourages efficient paths
        done = False

        if self._is_collision(new_head):
            reward = -10.0
            done = True
            self.snake.appendleft(new_head)
            return self._get_state(), reward, done, {"score": self.score}

        self.snake.appendleft(new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self.steps_since_food > self.max_steps_without_food:
            # Stop episodes where the snake just wanders forever
            reward = -10.0
            done = True

        return self._get_state(), reward, done, {"score": self.score}

    # ------------------------------------------------------------------ #
    # Rendering (used for screenshots / GIFs, not needed for training)
    # ------------------------------------------------------------------ #
    def render_grid(self):
        """Returns a (grid_size, grid_size) int array: 0=empty, 1=body, 2=head, 3=food."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i, (x, y) in enumerate(self.snake):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y, x] = 2 if i == 0 else 1
        fx, fy = self.food
        grid[fy, fx] = 3
        return grid

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _turn(self, action):
        idx = CLOCKWISE.index(self.direction)
        if action == 1:  # right turn
            idx = (idx + 1) % 4
        elif action == 2:  # left turn
            idx = (idx - 1) % 4
        # action == 0 keeps idx the same (straight)
        self.direction = CLOCKWISE[idx]

    def _place_food(self):
        occupied = set(self.snake)
        free_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        self.food = free_cells[np.random.randint(len(free_cells))]

    def _is_collision(self, point):
        x, y = point
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if point in list(self.snake)[:-1]:  # tail cell will move away, ignore it
            return True
        return False

    def _get_state(self):
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        idx = CLOCKWISE.index(self.direction)

        straight_dir = CLOCKWISE[idx]
        right_dir = CLOCKWISE[(idx + 1) % 4]
        left_dir = CLOCKWISE[(idx - 1) % 4]

        point_straight = (head_x + straight_dir[0], head_y + straight_dir[1])
        point_right = (head_x + right_dir[0], head_y + right_dir[1])
        point_left = (head_x + left_dir[0], head_y + left_dir[1])

        danger_straight = self._is_collision(point_straight)
        danger_right = self._is_collision(point_right)
        danger_left = self._is_collision(point_left)

        food_x, food_y = self.food

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(self.direction == LEFT),
            int(self.direction == RIGHT),
            int(self.direction == UP),
            int(self.direction == DOWN),
            int(food_x < head_x),
            int(food_x > head_x),
            int(food_y < head_y),
            int(food_y > head_y),
        ]
        return np.array(state, dtype=np.float32)
