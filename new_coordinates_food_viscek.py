import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
speed = 0.3
L = 32.0  # System size
rho = 3.0  # Particle density
N = int(rho*100)  # Number of particles

# Particle movement parameters
r0 = 2 # Interaction radius
deltat = 1.0  # Time step (unused)
factor = 0.5
v0 = r0/deltat*factor  # Base speed
eta = 0.1  # Noise parameter

# Food parameters
num_food = 10  # number of food sources
eat_radius = 0.5  # distance at which food is eaten
food_attraction_strength = 1.0  # strength of attraction to food

# Initialize particles
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)

# Initialize food positions
food_positions = np.random.uniform(0, L, size=(num_food, 2))

fig, ax = plt.subplots(figsize=(6, 6))
cos0, sin0 = np.cos(orient), np.sin(orient)
qv = ax.quiver(pos[:, 0], pos[:, 1], cos0, sin0, orient, clim=[-np.pi, np.pi])
sc = ax.scatter(food_positions[:, 0], food_positions[:, 1], marker='o', s=100, color='red')


def animate(i):
    global orient, pos, food_positions

    # Neighbor interactions
    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type='coo_matrix')
    data = np.exp(1j * orient[dist.col])
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

    # Food attraction: find closest food for each bird
    tree_food = cKDTree(food_positions, boxsize=[L, L])
    d_food, idx_food = tree_food.query(pos, k=1)
    # Compute periodic vector to food
    f_pos = food_positions[idx_food]
    delta = (f_pos - pos + L/2) % L - L/2
    norms = np.linalg.norm(delta, axis=1)
    unit_food = delta / norms[:, None]

    # Combine neighbor alignment and food attraction
    S_total = S + food_attraction_strength * (unit_food[:, 0] + 1j * unit_food[:, 1])

    # Update orientations with noise
    orient = np.angle(S_total) + eta * np.random.uniform(-np.pi, np.pi, size=N)

    # Update positions
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0 * speed
    pos[:, 1] += sin * v0 * speed
    pos %= L

    # Handle eating: respawn eaten food at random new locations
    eaten = np.unique(idx_food[d_food < eat_radius])
    for idx in eaten:
        food_positions[idx] = np.random.uniform(0, L, size=2)

    # Update plots
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, orient)
    sc.set_offsets(food_positions)
    return qv, sc

anim = FuncAnimation(fig, animate, frames=200, interval=10, blit=True)
plt.show()
