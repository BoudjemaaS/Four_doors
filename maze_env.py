
import gymnasium as gym
from maze import *
import random

class MazeEnv(gym.Env):
    """ Environnement pour labyrinthe"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, m, render_mode=None):
        super(MazeEnv, self).__init__()
        self.maze = m

        # On a 4 actions (haut, bas, gauche, droite)
        self.action_space = [Directions.UP, Directions.DOWN, Directions.LEFT, Directions.RIGHT]

        # Initialisation de la récompense
        self.reward = 0
        self.steps = 0
        # On positionne l'agent sur la case de départ
        self.current_state = self.maze.get_start()
        self.terminated = False
        self.truncated = False
        # Taille de la figure
        self.figsize = (10, 10)


    def get_random_action(self):
        return self.action_space[random.randint(0, 3)]

    def set_current_state(self, state):
        self.current_state = state
        self.maze.visit(state[0], state[1])

    def reset(self):
        self.set_current_state(self.maze.get_start())
        self.steps = 0
        self.truncated = False
        self.terminated = False
        return self.current_state

    def step(self, action):
        assert action in self.action_space, "Action not in action space!"
        next_state = self._move_agent(action)
        self.steps += 1

        # On verifie si l'agent a atteind une sortie
        if self.maze.is_terminal(next_state):
            self.terminated = True

        return next_state, self.maze.get_reward(next_state), self.terminated, self.truncated, {}

    def render(self):
        # Calculer la taille de la case en unités de la figure
        case_width = self.figsize[0] / self.maze.n_rows
        case_height = self.figsize[1] / self.maze.n_cols
        # Calculer la taille du texte en fonction de la taille de la case
        self.font_size = min(case_width, case_height) * 15
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        """Display the current state of the environment"""
        self.ax.clear()  # Erase the previous picture

        # Affiche la grille du labyrinthe
        self.ax.imshow(self.maze.array_types, cmap="Blues", interpolation="nearest", vmin=0,
                       vmax=1)  # Utilisation du colormap Greys
        for row in range(self.maze.n_rows):
            for col in range(self.maze.n_cols):
                text_color = "black"
                if self.maze.array_types[row][col] == CellTypes.EXIT.value:
                    self.ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='green', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == CellTypes.REGULAR.value:
                    self.ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='grey', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == CellTypes.WALL.value:
                    self.ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='red', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == CellTypes.START.value:
                    self.ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='lightgrey', alpha=0.5))
                    text_color = "white"

        self.ax.add_patch(plt.Circle((self.current_state[1],self.current_state[0]), 0.3, color=(0.9, 0.9, 0), fill=True))

        # Rendre l'affichage propre
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # Afficher l'animation
        plt.draw()
        # Attente pour permettre l'actualisation
        plt.pause(0.4)

    def _move_agent(self, action):
        """
        Move the agent from the current state to the next one following the action
        :param action: UP, DOWN, LEFT or RIGHT
        :return: the next state after applying the action. self.currrent_state is
        updated.
        """
        di, dj = action.value

        ni, nj = self.current_state[0] + di, self.current_state[1] + dj
        # On verifie si on est dans le labyrinthe
        if 0 <= ni < self.maze.n_rows and 0 <= nj < self.maze.n_cols:
            self.set_current_state((ni, nj))
        # sinon on ne bouge pas
        return self.current_state
