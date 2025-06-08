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

plt.rc('font', size=16) # Font size for plots

class CFG:
    """Configuration parameters for the simulation."""
    seed = 123

    # Simulation parameters
    L = 32.0  # System size
    N = 180
    VERTICAL_PLACEMENT_FACTOR = 2
    LEFT_HORIZONTAL_PLACEMENT_FACTOR = 2
    RIGHT_HORIZONTAL_PLACEMENT_FACTOR = 1.7

    # Duration
    NUM_FRAMES = 600

    # Particle movement
    W_ALIGN = 1.0
    R0 = 3
    v0 = 0.5
    ETA = 0.05

    # Initialization
    CLUSTERED = True

    # Food
    FOOD = False
    NF = 0
    EAT_RADIUS = 0.5
    FOOD_STRENGTH = 0.0
    RESPAWN_DELAY = 15

    # Metrics
    METRICS = True
    PLOT_METRICS = True

    # Timing
    START_DELAY = 3
    STOP_ON_FOOD_EATEN = True

    # Modes
    MODE = 3
    FOOD_STRENGTH_MIN = 5.0
    FOOD_STRENGTH_MAX = 10.0
    FOOD_STRENGTH_STEP = 1.0

    ETA_MIN = 0.01
    ETA_MAX = 0.20
    ETA_STEP = 0.01
    NF_VALUES = [1, 2, 4, 8]

    COHESION_FRAME_START = 100
    COHESION_FRAME_END = 800
    DEBUG = False

cfg = CFG()

def set_seed(seed=123):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

set_seed(cfg.seed)

def initialize_particles(cfg):
    """
    Initialize particle positions and orientations.
    Returns:
        pos (np.ndarray): Particle positions.
        orient (np.ndarray): Particle orientations.
    """
    if cfg.CLUSTERED:
        center_offset = 8.0 if cfg.NF == 1 else -12.0
        cluster_center = (np.array([cfg.L / 2, cfg.L / 2 + center_offset])) % cfg.L
        pos = (cluster_center + np.random.normal(scale=1.0, size=(cfg.N, 2))) % cfg.L
    else:
        pos = np.random.uniform(0, cfg.L, size=(cfg.N, 2))
    orient = np.full(cfg.N, np.pi / 2)
    return pos, orient

def create_evenly_spaced_food(num_x, num_y, L):
    """
    Generate evenly spaced food positions on a grid.
    Args:
        num_x (int): Number of food sources along x-axis.
        num_y (int): Number of food sources along y-axis.
        L (float): Size of the system.
    Returns:
        np.ndarray: Array of food positions.
    """
    food_positions = []
    dx = L / (num_x + 1)
    dy = L / (num_y + 1)
    for i in range(1, num_x + 1):
        for j in range(1, num_y + 1):
            food_positions.append([i * dx, j * dy])
    return np.array(food_positions)

def run_simulation():
    """
    Run the simulation based on the current mode specified in the configuration.

    Modes:
        1 - Single simulation
        2 - Multiple food strength experiments
        4 - Noise vs. food sources experiments
    """
    if cfg.MODE == 1:
        run_single_simulation()
    elif cfg.MODE == 2:
        run_distance_food_experiments()
    elif cfg.MODE == 3:
        run_noise_food_experiments()

def run_single_simulation(food_strength_value=None, eta_value=None, nf_value=None, show_animation=True):
    """Run a single simulation of flocking behavior.

        Args:
            food_strength_value (float): optional override for food strength.
            eta_value (float): optional override for noise.
            nf_value (int): optional override for number of food sources.
            show_animation (bool): whether to visualize the simulation.

        Returns:
            tuple: cohesion history and flock count history.
        """
    if food_strength_value is not None:
        current_food_strength = food_strength_value
    else:
        current_food_strength = cfg.FOOD_STRENGTH
    
    if eta_value is not None:
        current_eta = eta_value
    else:
        current_eta = cfg.ETA
    
    if nf_value is not None:
        current_nf = nf_value
    else:
        current_nf = cfg.NF
        
    # Set seed at the start of each simulation
    set_seed(cfg.seed)
    pos, orient = initialize_particles(cfg)

    cohesion_history = []
    flock_count_history = []
    
    if show_animation:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, cfg.L)
        ax.set_ylim(0, cfg.L)
        ax.set_aspect('equal') 
    else:
        fig, ax = None, None

    if cfg.FOOD:
        if current_nf == 1:
            # Place single food source in the center of the grid
            food_positions = np.array([[cfg.L / 2, cfg.L / 2]])
        elif current_nf == 2:
            offset = cfg.L * 0.25
            center_y = cfg.L / cfg.VERTICAL_PLACEMENT_FACTOR
            food_positions = np.array([
                [cfg.L / cfg.LEFT_HORIZONTAL_PLACEMENT_FACTOR - offset, center_y],
                [cfg.L / cfg.RIGHT_HORIZONTAL_PLACEMENT_FACTOR + offset, center_y]
            ])
        else:
            grid_side = int(np.ceil(np.sqrt(current_nf)))
            food_positions = create_evenly_spaced_food(grid_side, grid_side, cfg.L)[:current_nf]
        food_active = np.ones(current_nf, dtype=bool)
        food_timers = np.zeros(current_nf, dtype=int)
        food_lifetime = np.zeros(current_nf, dtype=int)
        food_creation_frame = np.zeros(current_nf, dtype=int)
        food_lifetime_history = []
        if show_animation:
            sc = ax.scatter(food_positions[food_active, 0], food_positions[food_active, 1], marker='o', s=100, color='red')
    else:
        if show_animation:
            sc = ax.scatter([], [], marker='o', s=100, color='red')

    if show_animation:
        cos0, sin0 = np.cos(orient), np.sin(orient)
        qv = ax.quiver(pos[:, 0], pos[:, 1], cos0, sin0, orient, clim=[-np.pi, np.pi])
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

    # Initialize timing variables for real-time delay
    start_time = time.time() if show_animation else None
    simulation_started = False
    simulation_stopped = False

    def simulate_frame(i):
        nonlocal pos, orient, food_active, food_timers, food_lifetime, food_creation_frame, food_lifetime_history, simulation_started, simulation_stopped
        
        # Check if we're still in the delay period (using real time)
        if show_animation and start_time is not None:
            elapsed_time = time.time() - start_time
            if elapsed_time < cfg.START_DELAY:
                remaining_time = cfg.START_DELAY - elapsed_time
                time_text.set_text(f'Starting in: {remaining_time:.1f}s')
                return qv, sc, time_text
        
        # For consistency, always use the actual simulation frame number
        # In Mode 1: subtract delay_frames, In Mode 2: use i directly
        delay_frames = int(cfg.START_DELAY * 100) if show_animation else 0
        sim_frame = i - delay_frames if show_animation and i >= delay_frames else (i if not show_animation else 0)
        
        # Check if simulation has been stopped
        if simulation_stopped:
            if show_animation:
                time_text.set_text('Simulation stopped - Food eaten!')
                return qv, sc, time_text
            return

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
                S_total += current_food_strength * food_vec

                global_indices = np.flatnonzero(food_active)[idx_food]
                eaten = np.unique(global_indices[d_food < cfg.EAT_RADIUS])
                for food_idx in eaten:
                    food_lifetime_history.append(food_lifetime[food_idx])
                    food_lifetime[food_idx] = 0
                    food_creation_frame[food_idx] = sim_frame
                food_active[eaten] = False
                food_timers[eaten] = 0

                # Check if simulation should stop when food is eaten
                if cfg.STOP_ON_FOOD_EATEN and len(eaten) > 0:
                    simulation_stopped = True
                    if show_animation:
                        time_text.set_text('Simulation stopped - Food eaten!')
                        return qv, sc, time_text
                    return

            food_lifetime[food_active] += 1
            food_timers[~food_active] += 1

            respawned = (food_timers >= cfg.RESPAWN_DELAY) & (~food_active)
            if np.any(respawned):
                food_active[respawned] = True
                food_timers[respawned] = 0
                food_creation_frame[respawned] = sim_frame

            if show_animation:
                sc.set_offsets(food_positions[food_active])

        orient = np.angle(S_total) + current_eta * np.random.uniform(-np.pi, np.pi, size=cfg.N)
        cos, sin = np.cos(orient), np.sin(orient)
        pos[:, 0] += cos * cfg.v0
        pos[:, 1] += sin * cfg.v0
        pos %= cfg.L

        if show_animation:
            qv.set_offsets(pos)
            qv.set_UVC(cos, sin, orient)
            time_text.set_text(f'Frame: {sim_frame}')

        if cfg.DEBUG and sim_frame % 100 == 0:
            print(f"Frame {sim_frame} took {time.perf_counter() - start:.4f}s")

        if cfg.METRICS:
            mean_heading = np.mean(np.exp(1j * orient))
            cohesion = np.abs(mean_heading)
            cohesion_history.append(cohesion)
            neighbor_graph = sparse.coo_matrix((np.ones_like(dist.data), (dist.row, dist.col)), shape=(cfg.N, cfg.N))
            n_components, labels = connected_components(neighbor_graph, directed=False)
            flock_count_history.append(n_components)

        if show_animation:
            return qv, sc, time_text

    if show_animation:
        simulate_frame(0)
        # For animation, we need enough frames to cover the delay + simulation
        delay_frames = int(cfg.START_DELAY * 100)
        total_frames = cfg.NUM_FRAMES + delay_frames
        anim = FuncAnimation(fig, simulate_frame, frames=total_frames, interval=10, blit=True, repeat=False, cache_frame_data=False)
        plt.show()
    else:
        # Run simulation without animation
        for i in range(cfg.NUM_FRAMES):
            simulate_frame(i)
            if simulation_stopped:
                break

    return cohesion_history, flock_count_history

def run_distance_food_experiments():
    """Run multiple simulations varying food strength and plot cohesion over time."""
    print("Running multiple experiments...")
    
    # Generate food strength values
    food_strengths = np.arange(cfg.FOOD_STRENGTH_MIN, cfg.FOOD_STRENGTH_MAX + cfg.FOOD_STRENGTH_STEP, cfg.FOOD_STRENGTH_STEP)
    
    all_cohesion_histories = {}
    
    for fs in food_strengths:
        print(f"Running experiment with FOOD_STRENGTH = {fs}")
        # Use the same seed for each experiment to ensure they start identically
        # Only the food strength changes between experiments
        cohesion_history, _ = run_single_simulation(food_strength_value=fs, show_animation=False)
        all_cohesion_histories[fs] = cohesion_history
    

    plt.figure(figsize=(12, 8))
    distinct_colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    for i, fs in enumerate(food_strengths):
        cohesion_data = all_cohesion_histories[fs]
        frames = np.arange(len(cohesion_data))
        color = distinct_colors[i % len(distinct_colors)]  
        
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
        line_style = line_styles[i % len(line_styles)]
        
        plt.plot(frames, cohesion_data, label=f'FS = {fs}', color=color, 
                linewidth=2.0, linestyle=line_style, alpha=0.8)
    
    plt.xlabel('Frame')
    plt.ylabel('Cohesion')
    plt.title(f'Cohesion Over Time - Multiple Food Strengths (NF = {cfg.NF})')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = f"cohesion_graph-NF_{cfg.NF}-FS_{cfg.FOOD_STRENGTH_MIN}-{cfg.FOOD_STRENGTH_MAX}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()

def run_noise_food_experiments():
    """Run simulations varying noise level and food source count, then plot average cohesion."""
    print("Running noise vs food sources experiments...")
    
    eta_values = np.arange(cfg.ETA_MIN, cfg.ETA_MAX + cfg.ETA_STEP, cfg.ETA_STEP)
    avg_cohesion_data = {}
    
    for nf in cfg.NF_VALUES:
        print(f"Testing with NF = {nf} food sources...")
        avg_cohesions = []
        
        for eta in eta_values:
            print(f"  Running with ETA = {eta:.3f}")
            
            # Run simulation with current parameters
            cohesion_history, _ = run_single_simulation(
                eta_value=eta, 
                nf_value=nf, 
                show_animation=False
            )
            
            # Calculate average cohesion from specified frame range
            start_frame = min(cfg.COHESION_FRAME_START, len(cohesion_history))
            end_frame = min(cfg.COHESION_FRAME_END, len(cohesion_history))
            
            if end_frame > start_frame:
                avg_cohesion = np.mean(cohesion_history[start_frame:end_frame])
            else:
                # If simulation is too short, use all available data
                avg_cohesion = np.mean(cohesion_history)
            
            avg_cohesions.append(avg_cohesion)
        
        avg_cohesion_data[nf] = avg_cohesions
    
    plt.figure(figsize=(12, 8))
    distinct_colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    for i, nf in enumerate(cfg.NF_VALUES):
        color = distinct_colors[i % len(distinct_colors)]  
        plt.plot(eta_values, avg_cohesion_data[nf], 
                label=f'NF = {nf}', color=color, 
                linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Noise Level (ETA)')
    plt.ylabel('Average Cohesion')
    plt.title(f'Average Cohesion vs Noise Level for Different Food Source Numbers\n(Averaged over frames {cfg.COHESION_FRAME_START}-{cfg.COHESION_FRAME_END})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    filename = f"noise_vs_cohesion-ETA_{cfg.ETA_MIN}-{cfg.ETA_MAX}-NF_{'-'.join(map(str, cfg.NF_VALUES))}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()
    
    return avg_cohesion_data

if __name__ == "__main__":
    run_simulation()