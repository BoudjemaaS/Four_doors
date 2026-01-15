import gymnasium as gym
import minigrid
import numpy as np
from torch import mode
from maze import *
from collections import defaultdict
from minigrid.envs.fourrooms import FourRoomsEnv
import pandas as pd


def epsilon_greedy_policy(state, Q, epsilon,n_actions):
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
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])  # Action gloutonne




#maze_env = FourRoomsEnv(9, max_steps=500, render_mode="rgb_array")

# =========================================================================================
# APPRENTISSAGE DE POLITIQUE VIA SARSA
# =========================================================================================
# Paramètres SARSA
alpha = 0.1    # Taux d'apprentissage
gamma = 0.9    # Facteur d'atténuation
epsilon = 0.1 # Probabilité d'exploration
epsilon_min = 0.05
episodes = 1000
epsilon_decay = (epsilon - epsilon_min) / episodes  # Décroissance de epsilon sur les épisodes
n_actions = 3
penalite = 0.02
max_steps = 1500
num_target=0

# Initialiser la table Q
Q=defaultdict(lambda: np.zeros(n_actions))
rewards_per_episode = []
step_per_episode = []
success_rate = 0
seed = 42
visit_counts = defaultdict(int)

maze_env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array", max_steps=max_steps)


# Entraînement avec SARSA
for episode in range(episodes):

    obs, info = maze_env.reset(seed=seed)
    direction = maze_env.unwrapped.agent_dir
    state = (maze_env.unwrapped.agent_pos[0], maze_env.unwrapped.agent_pos[1], maze_env.unwrapped.agent_dir)
    print("episode",episode)
    if episode == episodes - 1:
        last_episode_path = [(state[0], state[1])]

    action = epsilon_greedy_policy(state, Q, epsilon, n_actions)
    total_reward = 0
    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        step += 1
        visit_counts[(state[0], state[1])] += 1
        #print("step", step)
        next_obs, reward, terminated, truncated, info = maze_env.step(action)
        next_state = (maze_env.unwrapped.agent_pos[0], maze_env.unwrapped.agent_pos[1], maze_env.unwrapped.agent_dir)

        if episode == episodes - 1:
            last_episode_path.append((next_state[0], next_state[1]))

        reward -= penalite
        next_action = epsilon_greedy_policy(next_state, Q, epsilon, n_actions)

        # Mettre à jour Q selon l'équation SARSA
        target = reward + gamma * Q[next_state][next_action]
        Q[state][action] += alpha * (target - Q[state][action])

        # Passer à l'état suivant
        state = next_state
        action = next_action
        total_reward += reward + (1 - 0.9 * (step / max_steps))



    step_per_episode.append(step)
    rewards_per_episode.append(total_reward)
    if terminated: 
        success_rate+=1 
    epsilon = max(epsilon - epsilon_decay, epsilon_min)
maze_env.close()




# Build the optimal policy
optimal_policy = np.full((maze_env.unwrapped.grid.height, maze_env.unwrapped.grid.width), None, dtype=object)
# Initialisation de la politique

action_to_str = {0: "←", 1: "→", 2: "↑"} 

# On parcourt la table Q pour extraire la meilleure action par état
for state, actions_values in Q.items():
    x, y,d = state
    max_value = np.max(actions_values)
    
    if np.any(actions_values != 0): # Si l'agent a appris quelque chose sur cette case
        best_actions = [a for a, v in enumerate(actions_values) if v == max_value]
        # On stocke les symboles des meilleures actions
        optimal_policy[x, y] = "".join([action_to_str[a] for a in best_actions])



import matplotlib.pyplot as plt

def plot_optimal_policy():
    maze_env.reset(seed=seed)
    img = maze_env.render()
    width = maze_env.unwrapped.grid.width
    height = maze_env.unwrapped.grid.height
    
    plt.figure(figsize=(10, 10))
    
    
    path_x, path_y = zip(*last_episode_path)
    # On ajoute 0.5 pour centrer le point dans la case
    path_x = np.array(path_x) + 0.5
    path_y = np.array(path_y) + 0.5
    
    # Dessiner la ligne du chemin
    plt.plot(path_x, path_y, color='cyan', linewidth=2, label="Chemin de l'agent", alpha=0.8)
    # Marquer le départ et l'arrivée
    plt.scatter(path_x[0], path_y[0], color='yellow', s=100, label="Départ", zorder=5)
    plt.scatter(path_x[-1], path_y[-1], color='magenta', s=100, label="Fin", zorder=5)

    plt.imshow(img, extent=[0, width, height, 0])
    
    # On crée un dictionnaire pour regrouper les meilleures actions par case (x, y)
    best_per_cell = {} 

    for state, values in Q.items():
        x, y, d = state
        max_q_for_this_state = np.max(values)
        
        # Si on n'a pas encore de donnée pour cette case ou si cet état est meilleur
        if (x, y) not in best_per_cell or max_q_for_this_state > best_per_cell[(x, y)][0]:
            best_per_cell[(x, y)] = (max_q_for_this_state, np.argmax(values), d)

    # On dessine uniquement le meilleur état trouvé pour chaque case
    for (x, y), (q_val, action, direction) in best_per_cell.items():
        if q_val != 0: # On n'affiche que si l'agent a appris quelque chose
            if action == 2: # AVANCER
                dx, dy = 0, 0
                if direction == 0: dx = 0.4  # Droite
                elif direction == 1: dy = 0.4  # Bas
                elif direction == 2: dx = -0.4 # Gauche
                elif direction == 3: dy = -0.4 # Haut
                plt.arrow(x + 0.5, y + 0.5, dx, dy, head_width=0.2, color='red', alpha=0.8)
            else: # ROTATION
                color = 'blue' if action == 0 else 'green'
                plt.plot(x + 0.5, y + 0.5, marker='o', color=color, markersize=6)
    
    plt.title("Meilleure Action par Case (Basée sur la plus haute valeur Q)")

    plt.axis('off')
    plt.show()



def plot_map_visits():

    maze_env.reset(seed=seed)
    img = maze_env.render()
    width = maze_env.unwrapped.grid.width
    height = maze_env.unwrapped.grid.height
    
    plt.figure(figsize=(10, 10))
    
    path_x, path_y = zip(*last_episode_path)
    # On ajoute 0.5 pour centrer le point dans la case
    path_x = np.array(path_x) + 0.5
    path_y = np.array(path_y) + 0.5
    
    # Dessiner la ligne du chemin
    plt.plot(path_x, path_y, color='cyan', linewidth=2, label="Chemin de l'agent", alpha=0.8)
    # Marquer le départ et l'arrivée
    plt.scatter(path_x[0], path_y[0], color='yellow', s=100, label="Départ", zorder=5)
    plt.scatter(path_x[-1], path_y[-1], color='magenta', s=100, label="Fin", zorder=5)
    
    visit_matrix = np.zeros((height, width))
    for (x, y), count in visit_counts.items():
        visit_matrix[y, x] = count

    plt.imshow(img, extent=[0, width, height, 0], alpha=0.3)
    im = plt.imshow(visit_matrix, cmap='YlOrRd', extent=[0, width, height, 0], alpha=0.7)
    plt.colorbar(im, label="Nombre de visites")
    plt.title("Carte de chaleur des visites")
    plt.axis('off')
    plt.show()


def plot_rewards(axe):
    axe.plot(rewards_per_episode, label='Récompense par épisode')
    smoothed = pd.Series(rewards_per_episode).ewm(span=50).mean()
    axe.plot(smoothed, label='Moyenne mobile', color='red')
    axe.set_xlabel('Épisode')
    axe.set_ylabel('Récompense')
    axe.set_title('Récompense par épisode au fil du temps')
    

def plot_steps(axe):
    axe.plot(step_per_episode, label='Nombre de pas par épisode', color='orange')
    smoothed = pd.Series(step_per_episode).ewm(span=50).mean()
    axe.plot(smoothed, label='Moyenne mobile', color='red')
    axe.set_xlabel('Épisode')
    axe.set_ylabel('Nombre de pas')
    axe.set_title('Nombre de pas par épisode au fil du temps')
    




# Utilisation

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))

plot_rewards(axes[0])
plot_steps(axes[1])

plt.tight_layout() # Évite les chevauchements
plt.show()

plot_map_visits()
print("Taux de réussite sur", episodes, "épisodes :", (success_rate / episodes)*100 , "%")