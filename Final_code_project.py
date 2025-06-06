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
    FOOD = True
    NF = 2  # number of food sources 
    EAT_RADIUS = 0.5
    FOOD_STRENGTH = 100000.0
    RESPAWN_DELAY = 50 # = 0.5 second. This is the respawn of the food

    # Metrics
    METRICS = True
    PLOT_METRICS = False

    # Start delay (in seconds) - Fixed to be actual seconds
    START_DELAY = 3  # Add this parameter to control delay

    # Stop simulation when food is eaten
    STOP_ON_FOOD_EATEN = True  # Set to False to continue simulation after food is eaten

    # Experiment modes
    MODE = 2  # 1 = Single simulation, 2 = Multiple experiments, 4 = Noise vs Food Sources
    
    # Mode 2 parameters (for automated experiments)
    FOOD_STRENGTH_MIN = 100.0
    FOOD_STRENGTH_MAX = 1000.0
    FOOD_STRENGTH_STEP = 200.0

    # Mode 4 parameters (for noise vs food sources experiments)
    ETA_MIN = 0.01
    ETA_MAX = 0.20
    ETA_STEP = 0.01
    NF_VALUES = [1, 2, 4, 8]  # Different numbers of food sources to test
    
    # Frame range for averaging cohesion in Mode 4
    COHESION_FRAME_START = 100
    COHESION_FRAME_END = 800

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
            food_center = np.array([cfg.L / 2, cfg.L / 2])
            offset_above = -12.0  # vertical offset
            cluster_center = (food_center + np.array([0.0, offset_above])) % cfg.L
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

def run_single_simulation(food_strength_value=None, eta_value=None, nf_value=None, show_animation=True):
    """Run a single simulation with given parameters"""
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
        
    # Ensure consistent seeding - set seed at the start of each simulation
    set_seed(cfg.seed)
    pos, orient = initialize_particles(cfg)

    cohesion_history = []
    flock_count_history = []
    
    if show_animation:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, cfg.L)
        ax.set_ylim(0, cfg.L)
    else:
        fig, ax = None, None

    if cfg.FOOD:
        if current_nf == 1:
            # Place single food source in the center of the grid
            food_positions = np.array([[cfg.L / 2, cfg.L / 2]])
        elif current_nf == 2:
            offset = cfg.L * 0.25
            center_y = cfg.L / 2
            food_positions = np.array([
                [cfg.L / 2 - offset, center_y],
                [cfg.L / 2 + offset, center_y]
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


def run_multiple_experiments():
    """Run multiple experiments with different food strengths"""
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
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define easily distinguishable colors
    distinct_colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    for i, fs in enumerate(food_strengths):
        cohesion_data = all_cohesion_histories[fs]
        frames = np.arange(len(cohesion_data))
        color = distinct_colors[i % len(distinct_colors)]  # Cycle through colors if more lines than colors
        plt.plot(frames, cohesion_data, label=f'FS = {fs}', color=color, linewidth=1.5)
    
    plt.xlabel('Frame')
    plt.ylabel('Cohesion')
    plt.title(f'Cohesion Over Time - Multiple Food Strengths (NF = {cfg.NF})')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    filename = f"cohesion_graph-NF_{cfg.NF}-FS_{cfg.FOOD_STRENGTH_MIN}-{cfg.FOOD_STRENGTH_MAX}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()


def run_noise_food_experiments():
    """Run experiments testing different noise levels for different numbers of food sources"""
    print("Running noise vs food sources experiments...")
    
    # Generate ETA (noise) values
    eta_values = np.arange(cfg.ETA_MIN, cfg.ETA_MAX + cfg.ETA_STEP, cfg.ETA_STEP)
    
    # Dictionary to store average cohesion for each NF value
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
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define easily distinguishable colors for different NF values
    distinct_colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    for i, nf in enumerate(cfg.NF_VALUES):
        color = distinct_colors[i % len(distinct_colors)]  # Cycle through colors if more lines than colors
        plt.plot(eta_values, avg_cohesion_data[nf], 
                label=f'NF = {nf}', color=color, 
                linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Noise Level (ETA)')
    plt.ylabel('Average Cohesion')
    plt.title(f'Average Cohesion vs Noise Level for Different Food Source Numbers\n(Averaged over frames {cfg.COHESION_FRAME_START}-{cfg.COHESION_FRAME_END})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    filename = f"noise_vs_cohesion-ETA_{cfg.ETA_MIN}-{cfg.ETA_MAX}-NF_{'-'.join(map(str, cfg.NF_VALUES))}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()
    
    return avg_cohesion_data


def verify_modes_consistency():
    """Verify that Mode 1, Mode 2, and Mode 4 give identical results"""
    print("Verifying consistency between modes...")
    
    # Test with current configuration values
    test_fs = cfg.FOOD_STRENGTH
    test_eta = cfg.ETA
    test_nf = cfg.NF
    
    print(f"Testing with: FOOD_STRENGTH={test_fs}, ETA={test_eta}, NF={test_nf}")
    print("-" * 60)
    
    # Run Mode 1 simulation (without animation for fair comparison)
    print("Running Mode 1 simulation...")
    set_seed(cfg.seed)
    cohesion_mode1, _ = run_single_simulation(show_animation=False)
    
    # Run Mode 2 simulation (single experiment with same parameters)
    print("Running Mode 2 simulation...")
    set_seed(cfg.seed)  # Reset seed
    cohesion_mode2, _ = run_single_simulation(food_strength_value=test_fs, show_animation=False)
    
    # Run Mode 4 simulation (single experiment with same parameters)
    print("Running Mode 4 simulation...")
    set_seed(cfg.seed)  # Reset seed
    cohesion_mode4, _ = run_single_simulation(
        food_strength_value=test_fs, 
        eta_value=test_eta, 
        nf_value=test_nf, 
        show_animation=False
    )
    
    # Compare Mode 1 vs Mode 2
    print("\n--- Mode 1 vs Mode 2 Comparison ---")
    mode1_vs_mode2_consistent = True
    if len(cohesion_mode1) == len(cohesion_mode2):
        max_diff_12 = np.max(np.abs(np.array(cohesion_mode1) - np.array(cohesion_mode2)))
        print(f"Maximum difference in cohesion values: {max_diff_12}")
        if max_diff_12 < 1e-10:
            print("✓ Mode 1 and Mode 2 are consistent!")
        else:
            print("✗ Mode 1 and Mode 2 give different results!")
            mode1_vs_mode2_consistent = False
    else:
        print(f"✗ Different number of frames: Mode1={len(cohesion_mode1)}, Mode2={len(cohesion_mode2)}")
        mode1_vs_mode2_consistent = False
    
    # Compare Mode 1 vs Mode 4
    print("\n--- Mode 1 vs Mode 4 Comparison ---")
    mode1_vs_mode4_consistent = True
    if len(cohesion_mode1) == len(cohesion_mode4):
        max_diff_14 = np.max(np.abs(np.array(cohesion_mode1) - np.array(cohesion_mode4)))
        print(f"Maximum difference in cohesion values: {max_diff_14}")
        if max_diff_14 < 1e-10:
            print("✓ Mode 1 and Mode 4 are consistent!")
        else:
            print("✗ Mode 1 and Mode 4 give different results!")
            mode1_vs_mode4_consistent = False
    else:
        print(f"✗ Different number of frames: Mode1={len(cohesion_mode1)}, Mode4={len(cohesion_mode4)}")
        mode1_vs_mode4_consistent = False
    
    # Compare Mode 2 vs Mode 4
    print("\n--- Mode 2 vs Mode 4 Comparison ---")
    mode2_vs_mode4_consistent = True
    if len(cohesion_mode2) == len(cohesion_mode4):
        max_diff_24 = np.max(np.abs(np.array(cohesion_mode2) - np.array(cohesion_mode4)))
        print(f"Maximum difference in cohesion values: {max_diff_24}")
        if max_diff_24 < 1e-10:
            print("✓ Mode 2 and Mode 4 are consistent!")
        else:
            print("✗ Mode 2 and Mode 4 give different results!")
            mode2_vs_mode4_consistent = False
    else:
        print(f"✗ Different number of frames: Mode2={len(cohesion_mode2)}, Mode4={len(cohesion_mode4)}")
        mode2_vs_mode4_consistent = False
    
    # Overall result
    print("\n" + "=" * 60)
    all_consistent = mode1_vs_mode2_consistent and mode1_vs_mode4_consistent and mode2_vs_mode4_consistent
    if all_consistent:
        print("✓ ALL MODES ARE CONSISTENT!")
        print("The implementation is verified across all modes.")
    else:
        print("✗ INCONSISTENCIES DETECTED!")
        print("Please check the implementation for differences between modes.")
    
    print("=" * 60)
    return all_consistent


def run_simulation():
    if cfg.MODE == 1:
        # Mode 1: Single simulation
        cohesion_history, flock_count_history = run_single_simulation()
        
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
    
    elif cfg.MODE == 2:
        # Mode 2: Multiple experiments
        run_multiple_experiments()
    
    elif cfg.MODE == 3:
        # Mode 3: Verification mode (hidden mode for testing)
        verify_modes_consistency()
    
    elif cfg.MODE == 4:
        # Mode 4: Noise vs Food Sources experiments
        run_noise_food_experiments()

if __name__ == "__main__":
    run_simulation()