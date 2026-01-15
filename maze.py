import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class Directions(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

def direction_to_int(direction: Directions) -> int:
    if direction == Directions.UP:
        return 0
    elif direction == Directions.DOWN:
        return 1
    elif direction == Directions.LEFT:
        return 2
    elif direction == Directions.RIGHT:
        return 3
    else:
        return 4
def int_to_direction(i:int) -> Directions:
    if i == 0:
        return Directions.UP
    elif i == 1:
        return Directions.DOWN
    elif i == 2:
        return Directions.LEFT
    else:
        return Directions.RIGHT
class CellTypes(Enum):
    REGULAR = 0
    EXIT = 1
    WALL = 2
    START = 3


class Maze:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        # dans chaque case, on stocke le reward associé au fait d'arriver dans
        # cette case
        self.array_rewards = np.full((self.n_rows, self.n_cols), -1)
        # type of cells
        self.array_types = np.zeros((self.n_rows, self.n_cols), dtype=int)
        # On stocke le nombre de fois que l'on passe dans une case (un état donc)
        self.nb_visits = np.zeros((self.n_rows, self.n_cols))
        self.font_size = 1
        self.start = None

    def get_start(self):
        assert self.start is not None
        return self.start

    def add_wall(self, x, y, penalty=-100):
        assert 0 <= x < self.n_rows
        assert 0 <= y < self.n_cols
        self.array_rewards[x][y] = penalty
        self.array_types[x][y] = CellTypes.WALL.value

    def add_exit(self, x, y, reward=10):
        assert 0 <= x < self.n_rows
        assert 0 <= y < self.n_cols
        self.array_rewards[x][y] = reward
        self.array_types[x][y] = CellTypes.EXIT.value

    def add_start(self, x, y):
        assert 0 <= x < self.n_rows
        assert 0 <= y < self.n_cols
        assert self.start is None
        self.start = (x, y)
        self.array_rewards[x][y] = -1
        v = CellTypes.START.value
        self.array_types[x][y] = CellTypes.START.value

    def get_reward(self, pos):
        assert 0 <= pos[0] < self.n_rows
        assert 0 <= pos[1] < self.n_cols
        return self.array_rewards[pos[0]][pos[1]]

    def get_type(self, x, y):
        assert 0 <= x < self.n_rows
        assert 0 <= y < self.n_cols
        return self.array_types[x][y]

    def is_regular(self, x, y):
        assert 0 <= x < self.n_rows
        assert 0 <= y < self.n_cols
        return self.array_types[x][y] == CellTypes.REGULAR.value

    def is_exit(self, x, y):
        assert 0 <= x < self.n_rows
        assert 0 <= y < self.n_cols
        return self.array_types[x][y] == CellTypes.EXIT.value

    def is_wall(self, x, y):
        assert 0 <= x < self.n_rows
        assert 0 <= y < self.n_cols
        return self.array_types[x][y] == CellTypes.WALL.value

    def is_start(self, x, y):
        assert 0 <= x < self.n_rows
        assert 0 <= y < self.n_cols
        return self.array_types[x][y] == CellTypes.START.value

    def is_terminal(self, xy):
        return self.is_exit(xy[0], xy[1]) or self.is_wall(xy[0], xy[1])

    def visit(self,x,y):
        self.nb_visits[x,y] +=1


class MazePlot:
    def __init__(self, maze):
        self.maze = maze

    def _init_plot(self):
        # Taille de la figure
        self.figsize = (10, 10)
        # Calculer la taille de la case en unités de la figure
        case_width = self.figsize[0] / self.maze.n_rows
        case_height = self.figsize[1] / self.maze.n_cols

        # Calculer la taille du texte en fonction de la taille de la case
        self.font_size = min(case_width, case_height) * 15


    def plot(self):
        self._init_plot()
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.suptitle("Rewards", fontsize=16)

        ax.imshow(self.maze.array_types, cmap="Blues", interpolation="nearest", vmin=0,
                       vmax=1)

        for row in range(self.maze.n_rows):
            for col in range(self.maze.n_cols):
                text_color = "black"
                if self.maze.array_types[row][col] == CellTypes.EXIT.value:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='green', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == CellTypes.REGULAR.value:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='grey', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == CellTypes.WALL.value:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='red', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == CellTypes.START.value:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='lightblue', alpha=0.5))
                    text_color = "white"

                ax.text(col, row, self.maze.array_rewards[row][col], color=text_color, fontsize=self.font_size,
                             ha="center",
                             va="center")

        ax.set_xticks([])
        ax.set_yticks([])

    def plot_nb_visits(self):
        """
        Plot the number of visits of each state
        """

        self._init_plot()
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.suptitle("Nb. visits", fontsize=16)

        ax.imshow(self.maze.array_types, cmap="Blues", interpolation="nearest", vmin=0,
                       vmax=1)
        for row in range(self.maze.n_rows):
            for col in range(self.maze.n_cols):
                text_color = "black"
                if self.maze.array_types[row][col] == 1:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='green', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == 2:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='red', alpha=0.5))
                    text_color = "white"
                ax.text(col, row, self.maze.nb_visits[row][col], color=text_color, fontsize=self.font_size,
                             ha="center",
                             va="center")

        ax.set_xticks([])
        ax.set_yticks([])
    def plot_ij(self):
        """
        Plot the current state of the maze (wit color for start, end and wall) and (i,j) coords
        """
        self._init_plot()
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.suptitle("Maze structure", fontsize=16)

        ax.imshow(self.maze.array_types, cmap="Blues", interpolation="nearest", vmin=0,
                       vmax=1)
        for row in range(self.maze.n_rows):
            for col in range(self.maze.n_cols):
                text_color = "black"
                if self.maze.array_types[row][col] == 1:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='green', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == 2:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='red', alpha=0.5))
                    text_color = "white"
                ax.text(col, row, "("+str(row)+","+str(col)+")", color=text_color, fontsize=self.font_size,
                             ha="center",
                             va="center")

        ax.set_xticks([])
        ax.set_yticks([])

    def plot_values(self, val):

        self._init_plot()
        fig, ax = plt.subplots(figsize=self.figsize)
        # Affiche la grille du labyrinthe
        fig.suptitle("Value function V(s)", fontsize=16)

        ax.imshow(self.maze.array_types, cmap="Blues", interpolation="nearest", vmin=0,
                       vmax=1)  # Utilisation du colormap Greys
        for row in range(self.maze.n_rows):
            for col in range(self.maze.n_cols):
                text_color = "black"
                if self.maze.array_types[row][col] == CellTypes.EXIT.value:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='green', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == CellTypes.WALL.value:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='red', alpha=0.5))
                    text_color = "white"
                elif self.maze.array_types[row][col] == CellTypes.START.value:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='lightgrey', alpha=0.5))
                    text_color = "white"

                ax.text(col, row, round(val[row, col], 1), color=text_color,
                             fontsize=self.font_size, ha="center",
                             va="center")

        ax.set_xticks([])
        ax.set_yticks([])
    def plot_policy(self, P):
        self._init_plot()
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.suptitle("Greedy policy", fontsize=16)

        # Affiche la grille du labyrinthe
        ax.imshow(self.maze.array_types, cmap="Blues", interpolation="nearest", vmin=0,
                           vmax=1)
        for i in range(self.maze.n_rows):
            for j in range(self.maze.n_cols):
                if self.maze.is_regular(i, j) or self.maze.is_start(i,j):
                    center = (i, j)
                    # Trouver toutes les clés ayant la valeur max
                    max_pairs = P[i,j]
                    if max_pairs is None:
                        max_pairs = [Directions.UP, Directions.DOWN, Directions.LEFT, Directions.RIGHT]
                    for pair in max_pairs:
                        # Dictionnaire des décalages selon la direction
                        direction_offsets = {
                            Directions.UP: (-0.5, 0),
                            Directions.DOWN: (0.5, 0),
                            Directions.LEFT: (0, -0.5),
                            Directions.RIGHT: (0, 0.5)
                        }
                        offset_y, offset_x = direction_offsets[pair]
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='grey', alpha=0.5))
                        ax.annotate('', xy=(center[1] + offset_x, center[0] + offset_y),
                                         xytext=(center[1], center[0]),
                                         arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=5))
                else:
                    text_color = "white"
                    if self.maze.array_types[i][j] == CellTypes.EXIT.value:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='green', alpha=0.5))
                    elif self.maze.array_types[i][j] == CellTypes.WALL.value:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='red', alpha=0.5))
                    elif self.maze.array_types[i][j] == CellTypes.START.value:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightgrey', alpha=0.5))


        ax.set_xticks([])
        ax.set_yticks([])

    def show(self):
        plt.show()


# Fonction pour extraire la politique optimale
def extract_greedy_policy_from_V(maze, V):
    m, n = V.shape
    policy = np.full((m, n), None, dtype=object)  # Initialisation de la politique

    for i in range(m):
        for j in range(n):
            # Trouver la valeur maximale
            q_val = {}

            for action in [Directions.UP, Directions.DOWN, Directions.LEFT, Directions.RIGHT]:
                # calcul de la nouvelle position
                di, dj = action.value
                ni, nj = i + di, j + dj

                # Vérifier si la nouvelle position est dans les limites du labyrinthe
                if 0 <= ni < m and 0 <= nj < n and  not maze.is_wall(ni, nj):
                    q_val[action] = V[ni][nj]
                else:
                    q_val[action] = V[i][j]

            max_val = max(q_val.values())
            policy[i, j] = [key for key, value in q_val.items() if value == max_val]
    return policy



def extract_greedy_policy_from_Q(maze,Q):

    policy = np.full((maze.n_rows, maze.n_cols), None, dtype=object)  # Initialisation de la politique

    for state, action_values in Q.items():
        # Sélectionne l'indice de l'action avec la valeur maximale
        policy[state[0], state[1]] = [int_to_direction(np.argmax(action_values))]

    return policy
