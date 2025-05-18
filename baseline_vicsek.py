import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.csgraph import connected_components

np.random.seed(123) 

# Tracking variables initialization
cohesion_history = []
stabilization_threshold = 0.008 # change in cohesion to consider "stable"
stabilization_window = 100       # number of frames to consider for stabilization
stabilized_frame = None         # frame where stabilization is detected
flock_history = []
food_release_frame = None
post_food_cohesion = []
post_food_flock_counts = []

# Simulation parameters
speed = 0.3
L = 32.0  # System size
rho = 3.0  # Particle density
N = int(rho*30)  # Number of particles
print(N)

# Particle movement parameters
w_align = 1.0
w_cohesion = 0.02
w_separation = 0.01
separation_radius = 0.3
r0 = 2 # Interaction radius
deltat = 1.0  # Time step (unused)
factor = 0.5
v0 = r0/deltat*factor  # Base speed
eta = 0.05  # Noise parameter

# Food parameters
num_food = 0  # number of food sources (fixed positions)
eat_radius = 0.5
food_attraction_strength = 1.0

# Initialize particles
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)

# Create fixed food positions (uniformly spaced in a 2D grid)
grid_side = int(np.ceil(np.sqrt(num_food)))
x_coords = np.linspace(0.2 * L, 0.8 * L, grid_side)
y_coords = np.linspace(0.2 * L, 0.8 * L, grid_side)
xx, yy = np.meshgrid(x_coords, y_coords)
food_positions = np.vstack([xx.ravel(), yy.ravel()]).T[:num_food]

# Track which food sources are "active" (True = available)
food_active = np.ones(num_food, dtype=bool)
respawn_delay = 10  # frames (0.1 second at 10 ms/frame)
food_timers = np.zeros(num_food, dtype=int)
food_released = False # only release food spawn after flock has stabilized


fig, ax = plt.subplots(figsize=(6, 6))
cos0, sin0 = np.cos(orient), np.sin(orient)
qv = ax.quiver(pos[:, 0], pos[:, 1], cos0, sin0, orient, clim=[-np.pi, np.pi])
sc = ax.scatter(food_positions[:, 0], food_positions[:, 1], marker='o', s=100, color='red')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

def animate(i):
    global orient, pos, food_active, stabilized_frame, food_released, food_release_frame

    # Neighbor interactions
    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type='coo_matrix')

    # Precompute neighbor contributions
    orient_neighbors = orient[dist.col]
    pos_neighbors = pos[dist.col]
    pos_self = pos[dist.row]

    delta_pos = (pos_neighbors - pos_self + L / 2) % L - L / 2

    # ---------- ALIGNMENT ----------
    alignment_vec = np.exp(1j * orient_neighbors)
    alignment_matrix = sparse.coo_matrix((alignment_vec, (dist.row, dist.col)), shape=(N, N))
    alignment_sum = np.squeeze(np.asarray(alignment_matrix.tocsr().sum(axis=1)))

    # ---------- COHESION ----------
    cohesion_matrix = sparse.coo_matrix((delta_pos[:, 0] + 1j * delta_pos[:, 1], (dist.row, dist.col)), shape=(N, N))
    cohesion_vec = np.squeeze(np.asarray(cohesion_matrix.tocsr().sum(axis=1)))

    # ---------- SEPARATION ----------
    separation_force = -delta_pos / (np.linalg.norm(delta_pos, axis=1, keepdims=True) + 1e-8)
    mask = np.linalg.norm(delta_pos, axis=1) < separation_radius
    separation_force = separation_force[mask]
    sep_rows = dist.row[mask]
    separation_complex = separation_force[:, 0] + 1j * separation_force[:, 1]
    separation_matrix = sparse.coo_matrix((separation_complex, (sep_rows, sep_rows)), shape=(N, N))
    separation_vec = np.squeeze(np.asarray(separation_matrix.tocsr().sum(axis=1)))

    # Combine all behaviors
    S_total = (
        w_align * alignment_sum +
        w_cohesion * cohesion_vec +
        w_separation * separation_vec
    )

    # Release food only after both cohesion and flocking conditions are met
    if 'flocking_frame' in globals() and stabilized_frame is not None and not food_released:
        print(f"Food released at frame {i} ({i * 0.01:.2f} s)")
        food_released = True
        food_release_frame = i  # Track the frame food appears

    # Handle food behavior only after release
    if food_released and len(food_positions) > 0:
        active_food_positions = food_positions[food_active]

        tree_food = cKDTree(active_food_positions, boxsize=[L, L])
        d_food, idx_food = tree_food.query(pos, k=1)
        f_pos = active_food_positions[idx_food]
        delta_food = (f_pos - pos + L / 2) % L - L / 2
        norms = np.linalg.norm(delta_food, axis=1)
        unit_food = delta_food / norms[:, None]
        food_vec = unit_food[:, 0] + 1j * unit_food[:, 1]
        S_total += food_attraction_strength * food_vec

    # Update orientations with noise
    orient = np.angle(S_total) + eta * np.random.uniform(-np.pi, np.pi, size=N)

    # Update positions
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0 * speed
    pos[:, 1] += sin * v0 * speed
    pos %= L

    # Handle eating
    if food_released and len(food_positions) > 0:
        global_indices = np.flatnonzero(food_active)[idx_food]
        eaten = np.unique(global_indices[d_food < eat_radius])
        food_active[eaten] = False
        food_timers[eaten] = 0

        food_timers[~food_active] += 1
        respawned = (food_timers >= respawn_delay) & (~food_active)
        food_active[respawned] = True
        food_timers[respawned] = 0

    # Update plots
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, orient)

    if food_released:
        sc.set_offsets(food_positions[food_active])
    else:
        sc.set_offsets(np.empty((0, 2)))  

    if i == num_frames - 1:
        plt.close(fig)

    elapsed_time = i * 0.01
    time_text.set_text(f'Time: {elapsed_time:.2f} s')

    mean_heading = np.mean(np.exp(1j * orient))
    cohesion = np.abs(mean_heading)
    cohesion_history.append(cohesion)
    if food_released:
        post_food_cohesion.append(cohesion)

    if len(cohesion_history) > stabilization_window:
        recent = cohesion_history[-stabilization_window:]
        if max(recent) - min(recent) < stabilization_threshold and stabilized_frame is None:
            stabilized_frame = i
            print(f"Group cohesion stabilized at frame {i} ({elapsed_time:.2f} s) with cohesion ≈ {cohesion:.3f}")
    
    neighbor_graph = sparse.coo_matrix((np.ones_like(dist.data), (dist.row, dist.col)), shape=(N, N))
    n_components, labels = connected_components(neighbor_graph, directed=False)
    flock_history.append(n_components)

    if food_released:
        post_food_flock_counts.append(n_components)

    if n_components == 1 and 'flocking_frame' not in globals():
        global flocking_frame
        flocking_frame = i
        print(f"All boids formed one flock at frame {i} ({elapsed_time:.2f} s)")

    return qv, sc, time_text

num_frames = 2000
anim = FuncAnimation(fig, animate, frames=num_frames, interval=10, blit=True, repeat=False)
plt.show()

plt.figure()
if food_release_frame is not None:
    t = np.arange(food_release_frame) * 0.01
    plt.plot(t, cohesion_history[:food_release_frame])
else:
    t = np.arange(len(cohesion_history)) * 0.01
    plt.plot(t, cohesion_history)
plt.xlabel("Time (s)")
plt.ylabel("Cohesion")
plt.title("Group Cohesion (Before Food)")
plt.grid(True)
plt.show()

if food_release_frame is not None and len(post_food_cohesion) > 0:
    plt.figure()
    t_post = np.arange(len(post_food_cohesion)) * 0.01  # Time after food appears
    plt.plot(t_post, post_food_cohesion)
    plt.xlabel("Time After Food Appears (s)")
    plt.ylabel("Cohesion")
    plt.title("Group Cohesion (After Food)")
    plt.grid(True)
    plt.show()

plt.figure()
if food_release_frame is not None:
    t = np.arange(food_release_frame) * 0.01
    plt.plot(t, flock_history[:food_release_frame])
else:
    t = np.arange(len(flock_history)) * 0.01
    plt.plot(t, flock_history)
plt.xlabel("Time (s)")
plt.ylabel("Number of Flocks")
plt.title("Flock Count (Before Food)")
plt.grid(True)
plt.show()

if food_release_frame is not None and len(post_food_flock_counts) > 0:
    plt.figure()
    t_post = np.arange(len(post_food_flock_counts)) * 0.01  # Time after food appears
    plt.plot(t_post, post_food_flock_counts)
    plt.xlabel("Time After Food Appears (s)")
    plt.ylabel("Number of Flocks")
    plt.title("Flock Count (After Food)")
    plt.grid(True)
    plt.show()