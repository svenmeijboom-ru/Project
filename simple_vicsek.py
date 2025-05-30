import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.csgraph import connected_components
import random
import os
import time

class CFG:
    seed = 123

    # Simulation parameters
    L = 32.0  # System size
    N = 180

    # Simulation duration
    NUM_FRAMES = 2000

    # Particle movement parameters
    W_ALIGN = 1.0 # Weight for Vicsek alignment
    R0 = 3 # Interaction radius
    v0 = 0.5 #R0/DELTAT*FACTOR  # Base speed
    ETA = 0.05  # Noise parameter

    # Initialization of particles
    CLUSTERED = True

    # Food
    FOOD = False
    NF = 1  # number of food sources 
    EAT_RADIUS = 0.5
    FOOD_STRENGTH = 0.0
    RESPAWN_DELAY = 50 # = 0.5 second

    # Metrics
    METRICS = True
    PLOT_METRICS = True

    DEBUG = False

cfg = CFG()

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)

set_seed(cfg.seed)

def initialize_particles(cfg):
    if cfg.CLUSTERED:
        if cfg.NF == 1:
            food_center = np.array([cfg.L / 2, cfg.L / 2])
            offset_above = 8.0  # vertical offset
            cluster_center = (food_center + np.array([0.0, offset_above])) % cfg.L
        else:
            cluster_center = np.array([cfg.L / 2, cfg.L / 2])
        cluster_radius = 1.0  # Controls spread of initial cluster
        offsets = np.random.normal(scale=cluster_radius, size=(cfg.N, 2))
        pos = (cluster_center + offsets) % cfg.L
    else:
        pos = np.random.uniform(0, cfg.L, size=(cfg.N, 2))
    orient = np.random.uniform(-np.pi, np.pi, size=cfg.N)
    return pos, orient

def create_evenly_spaced_food(num_x, num_y, L):
    food_positions = []
    dx = L / (num_x + 1)
    dy = L / (num_y + 1)
    for i in range(1, num_x + 1):
        x = i * dx
        for j in range(1, num_y + 1):
            y = j * dy
            food_positions.append([x, y])
    return np.array(food_positions)

def run_simulation():
    set_seed(cfg.seed)
    pos, orient = initialize_particles(cfg)

    if cfg.METRICS:
        cohesion_history = []
        flock_count_history = []
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, cfg.L)
    ax.set_ylim(0, cfg.L)

    if cfg.FOOD:
        if cfg.NF == 2:
            offset = cfg.L * 0.25
            center_y = cfg.L / 2
            food_positions = np.array([
                [cfg.L / 2 - offset, center_y],
                [cfg.L / 2 + offset, center_y]
            ])
        else:
            grid_side = int(np.ceil(np.sqrt(cfg.NF)))
            food_positions = create_evenly_spaced_food(grid_side, grid_side, cfg.L)[:cfg.NF]
        food_active = np.ones(cfg.NF, dtype=bool)
        food_timers = np.zeros(cfg.NF, dtype=int)
        food_lifetime = np.zeros(cfg.NF, dtype=int)
        food_creation_frame = np.zeros(cfg.NF, dtype=int)
        food_lifetime_history = []
        sc = ax.scatter(food_positions[food_active, 0], food_positions[food_active, 1], marker='o', s=100, color='red')
    else:
        sc = ax.scatter([], [], marker='o', s=100, color='red')

    cos0, sin0 = np.cos(orient), np.sin(orient)
    qv = ax.quiver(pos[:, 0], pos[:, 1], cos0, sin0, orient, clim=[-np.pi, np.pi])
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

    def animate(i):
        nonlocal pos, orient, food_active, food_timers, food_lifetime, food_creation_frame, food_lifetime_history
        if cfg.DEBUG:
            start = time.perf_counter()
        tree = cKDTree(pos, boxsize=[cfg.L, cfg.L])
        dist = tree.sparse_distance_matrix(tree, max_distance=cfg.R0, output_type='coo_matrix')
        orient_neighbors = orient[dist.col]
        alignment_vec = np.exp(1j * orient_neighbors)
        alignment_matrix = sparse.coo_matrix((alignment_vec, (dist.row, dist.col)), shape=(cfg.N, cfg.N))
        alignment_sum = np.squeeze(np.asarray(alignment_matrix.tocsr().sum(axis=1)))
        S_total = cfg.W_ALIGN * alignment_sum

        if cfg.FOOD:
            active_food = food_positions[food_active]
            if len(active_food) > 0:
                tree_food = cKDTree(active_food, boxsize=[cfg.L, cfg.L])
                d_food, idx_food = tree_food.query(pos, k=1)
                f_pos = active_food[idx_food]
                delta_food = (f_pos - pos + cfg.L / 2) % cfg.L - cfg.L / 2
                norms = np.linalg.norm(delta_food, axis=1)
                unit_food = delta_food / (norms[:, None] + 1e-8)
                food_vec = unit_food[:, 0] + 1j * unit_food[:, 1]
                S_total += cfg.FOOD_STRENGTH * food_vec

                global_indices = np.flatnonzero(food_active)[idx_food]
                eaten = np.unique(global_indices[d_food < cfg.EAT_RADIUS])
                for food_idx in eaten:
                    food_lifetime_history.append(food_lifetime[food_idx])
                    food_lifetime[food_idx] = 0
                    food_creation_frame[food_idx] = i
                food_active[eaten] = False
                food_timers[eaten] = 0

            food_lifetime[food_active] += 1
            food_timers[~food_active] += 1

            respawned = (food_timers >= cfg.RESPAWN_DELAY) & (~food_active)
            if np.any(respawned):
                food_active[respawned] = True
                food_timers[respawned] = 0
                food_creation_frame[respawned] = i
            
            sc.set_offsets(food_positions[food_active])

        orient = np.angle(S_total) + cfg.ETA * np.random.uniform(-np.pi, np.pi, size=cfg.N)
        cos, sin = np.cos(orient), np.sin(orient)
        pos[:, 0] += cos * cfg.v0 
        pos[:, 1] += sin * cfg.v0 
        pos %= cfg.L

        qv.set_offsets(pos)
        qv.set_UVC(cos, sin, orient)
        time_text.set_text(f'Frame: {i}')

        if cfg.DEBUG and i % 100 == 0:
            print(f"Frame {i} took {time.perf_counter() - start:.4f}s")

        if cfg.METRICS:
            mean_heading = np.mean(np.exp(1j * orient))
            cohesion = np.abs(mean_heading)
            cohesion_history.append(cohesion)
            neighbor_graph = sparse.coo_matrix((np.ones_like(dist.data), (dist.row, dist.col)), shape=(cfg.N, cfg.N))
            n_components, labels = connected_components(neighbor_graph, directed=False)
            flock_count_history.append(n_components)

        return qv, sc, time_text
    
    animate(0)
    anim = FuncAnimation(fig, animate, frames=cfg.NUM_FRAMES, interval=10, blit=True, repeat=False, cache_frame_data=False)
    plt.show()

    if cfg.METRICS and cfg.PLOT_METRICS:
        frames = np.arange(len(cohesion_history))

        plt.figure(figsize=(10, 4))
        plt.plot(frames, cohesion_history, label='Cohesion')
        plt.xlabel('Frame')
        plt.ylabel('Cohesion')
        plt.title('Cohesion Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(frames, flock_count_history, label='Flock Count')
        plt.xlabel('Frame')
        plt.ylabel('Number of Flocks')
        plt.title('Flock Count Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    run_simulation()