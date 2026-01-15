from maze import *
import maze_env as env

# =========================================================================================
# FONCTION DE POLITIQUE EPSILON-GLOUTONNE
# =========================================================================================
def epsilon_greedy_policy(env, state, Q, epsilon):
    """
    Définition d'une politique de type Epsilon Greedy agissant sur la fonction
    de valeur état
    :param env: Environment sur lequel la politique s'appliqe
    :param state: Etat observé par l'agent
    :param Q:   Fonction de valeur état-action
    :param epsilon: paramètre pour contrôler le ration exploration/exploitation
    :return: l'action choisie par la politique depuis l'état 'state'
    """
    if np.random.rand() < epsilon:
        return env.action_space[np.random.choice(4)]  # Action aléatoire
    else:
        return env.action_space[np.argmax([Q[state][int_to_direction(a)] for a in range(4)])]  # Action gloutonne


# =========================================================================================
# DEFINITION DE L'ENVIRONNEMENT
# =========================================================================================
ENV_FALAISE = True

if ENV_FALAISE:
    maze = Maze(4, 12)
    maze.add_exit(3, 11, 10)
    maze.add_wall(3, 1)
    maze.add_wall(3, 2)
    maze.add_wall(3, 3)
    maze.add_wall(3, 4)
    maze.add_wall(3, 5)
    maze.add_wall(3, 6)
    maze.add_wall(3, 7)
    maze.add_wall(3, 8)
    maze.add_wall(3, 9)
    maze.add_wall(3, 10)
    maze.add_start(3, 0)
else:
    maze = Maze(10, 10)
    maze.add_exit(9, 9, 1000)
    maze.add_start(0, 0)

maze_env = env.MazeEnv(maze)


# =========================================================================================
# APPRENTISSAGE DE POLITIQUE VIA SARSA
# =========================================================================================
# Paramètres SARSA
alpha = 0.1    # Taux d'apprentissage
gamma = 0.9    # Facteur d'atténuation
epsilon = 0.1  # Probabilité d'exploration
episodes = 1000

# Initialiser la table Q
Q = {(i, j): {a: 0 for a in Directions} for i in range(maze.n_rows) for j in range(maze.n_cols)}
rewards_per_episode = []
success_rate = []

# Entraînement avec SARSA
for episode in range(episodes):
    state = maze_env.reset()
    action = epsilon_greedy_policy(maze_env,state, Q, epsilon)
    total_reward = 0
    successful = False
    while not maze.is_terminal(state):
        next_state, reward, _, _ ,_= maze_env.step(action)
        next_action = epsilon_greedy_policy(maze_env, next_state, Q, epsilon)

        # Mettre à jour Q selon l'équation SARSA
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )
        total_reward += reward
        if maze.is_terminal(next_state):
            successful = True

        # Passer à l'état suivant
        state = next_state
        action = next_action

    rewards_per_episode.append(total_reward)
    success_rate.append(1 if successful else 0)

# Build the optimal policy
optimal_policy = np.full((maze.n_rows, maze.n_cols), None, dtype=object)
# Initialisation de la politique

for state in Q:
    if maze.is_terminal(state):
        continue
    print(state, ", ",Q[state])
    max_value = max(Q[state].values())  # Trouver la valeur maximale
    best_actions = [action for action, value in Q[state].items() if value == max_value]
    optimal_policy[state] = set(best_actions)  # Stocker toutes les actions optimales comme un ensemble


# =========================================================================================
# TRACAGE DE COURBES
# =========================================================================================
# Tracer les courbes
plt.figure(figsize=(10, 5))

# Cumul des récompenses
plt.subplot(1, 1, 1)
plt.plot(rewards_per_episode, label="SARSA - Cumul des récompenses")
plt.xlabel("Épisode")
plt.ylabel("Cumul des récompenses")
plt.title("Cumul des récompenses par épisode")
plt.legend()

# Convertir le taux de réussite en une moyenne glissante
#success_rate_smoothed = np.convolve(success_rate, np.ones(50) / 50, mode='valid')
#
# # Taux de réussite
# plt.subplot(1, 2, 2)
# plt.plot(success_rate, label="Taux de réussite (moyenne glissante)", color="green")
# plt.xlabel("Épisode")
# plt.ylabel("Taux de réussite")
# plt.title("Évolution du taux de réussite")
# plt.legend()

plt.tight_layout()
plt.show()

# Afficher les valeurs estimées des états
mplt = MazePlot(maze)
mplt.plot()
mplt.plot_nb_visits()
mplt.plot_policy(optimal_policy)
mplt.show()
