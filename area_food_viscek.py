import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
L = 32.0
rho = 3.0
N = int(rho * L**2)
print(" N", N)

speed = 0.2

r0 = 1.0
deltat = 1.0
factor = 0.5
v0 = r0 / deltat * factor
eta = 0.15
iterations = 10000

# Particle state
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)

# Food field
grid_res = 64
food = np.random.rand(grid_res, grid_res)

def update_food(t):
    """Creates a dynamic, smooth food pattern."""
    x = np.linspace(0, 2 * np.pi, grid_res)
    y = np.linspace(0, 2 * np.pi, grid_res)
    X, Y = np.meshgrid(x, y)
    global food
    food = 0.5 + 0.5 * np.sin(X + t * 0.1) * np.cos(Y - t * 0.1)

# Set up visualization
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, L)
ax.set_ylim(0, L)

# Food visualization
im = ax.imshow(food, extent=[0, L, 0, L], origin='lower', cmap='Greens', alpha=0.5)

# Particle visualization
qv = ax.quiver(pos[:, 0], pos[:, 1], np.cos(orient), np.sin(orient), orient, clim=[-np.pi, np.pi])

def animate(i):
    global orient, pos

    update_food(i * speed)  # update food field
    im.set_data(food)

    # Find neighbors
    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type='coo_matrix')

    # Alignment
    data = np.exp(orient[dist.col] * 1j)
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

    # Local food sensing
    grid_x = (pos[:, 0] / L * grid_res).astype(int) % grid_res
    grid_y = (pos[:, 1] / L * grid_res).astype(int) % grid_res
    local_food = food[grid_y, grid_x]

    # Orientation update: alignment + food bias + noise
    food_bias_strength = 0.5
    orient = np.angle(S) + food_bias_strength * (local_food - 0.5) * np.pi + eta * np.random.uniform(-np.pi, np.pi, size=N)

    # Movement
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0 * speed
    pos[:, 1] += sin * v0 * speed
    pos[pos > L] -= L
    pos[pos < 0] += L

    # Update plot
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, orient)
    return qv, im

anim = FuncAnimation(fig, animate, np.arange(1, 200), interval=1, blit=True)
plt.show()
