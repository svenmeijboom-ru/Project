import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.csgraph import connected_components
import os
import time

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

# Flocking detection parameters
flocking_confirmation_window = 50  # Number of consecutive frames to confirm stable flocking
flocking_candidates = []           # Track recent frame numbers where n_components == 1
flocking_confirmed = False         # Whether stable flocking has been confirmed

# Time tracking
start_real_time = None  # Will be set when animation starts
frame_timestamps = []   # Real time when each frame was processed

# Simulation parameters
speed = 0.5
L = 32.0  # System size
rho = 3.0  # Particle density
N = int(rho*30)  # Number of particles
print(N)

# Particle movement parameters
w_align = 1.0      # Weight for Vicsek alignment
w_cohesion = 0.02   # Weight for Boids cohesion
w_separation = 0.01  # Weight for Boids separation
separation_radius = 0.5
r0 = 3 # Interaction radius
deltat = 1.0  # Time step (unused)
factor = 0.5
v0 = r0/deltat*factor  # Base speed
eta = 0.05  # Noise parameter

# Food parameters
num_food = 9  # number of food sources (fixed positions)
eat_radius = 1
food_attraction_strength = 1

# Food lifetime metrics
food_lifetime = np.zeros(num_food, dtype=int)  # Track frames each food exists
food_creation_frame = np.zeros(num_food, dtype=int)  # When each food was created/respawned (in frames)
food_lifetime_history = []  # Track the lifetime of each consumed food (in frames)
frame_to_consumption = []  # Frames from spawn to consumption
last_food_id = None  # Track the ID of the last food consumed
total_food_eaten = 0  # Track total food consumed in the experiment

# Initialize particles
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)


def create_evenly_spaced_food(num_x, num_y, L):
    """
    Create evenly spaced food sources in a grid pattern
    
    Parameters:
    num_x (int): Number of food sources along x-axis
    num_y (int): Number of food sources along y-axis
    L (float): System size
    
    Returns:
    numpy.ndarray: Array of food positions with shape (num_x*num_y, 2)
    """
    food_positions = []
    
    # Calculate spacing and offset
    dx = L / (num_x + 1)
    dy = L / (num_y + 1)
    
    # Generate grid positions
    for i in range(1, num_x + 1):
        x = i * dx
        for j in range(1, num_y + 1):
            y = j * dy
            food_positions.append([x, y])
    
    return np.array(food_positions)

# Then add this code to create the food positions:
# If you want a square grid (2x2, 3x3, etc.)
grid_side = int(np.ceil(np.sqrt(num_food)))
food_positions = create_evenly_spaced_food(grid_side, grid_side, L)[:num_food]


# Track which food sources are "active" (True = available)
food_active = np.ones(num_food, dtype=bool)
respawn_delay = 100  # frames
food_timers = np.zeros(num_food, dtype=int)
food_released = False # only release food spawn after flock has stabilized


fig, ax = plt.subplots(figsize=(6, 6))
cos0, sin0 = np.cos(orient), np.sin(orient)
qv = ax.quiver(pos[:, 0], pos[:, 1], cos0, sin0, orient, clim=[-np.pi, np.pi])
sc = ax.scatter(food_positions[:, 0], food_positions[:, 1], marker='o', s=100, color='red')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')
food_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10, color='blue')
real_time_text = ax.text(0.02, 0.8, '', transform=ax.transAxes, fontsize=10, color='gray')

def animate(i):
    global orient, pos, food_active, stabilized_frame, food_released, food_release_frame
    global food_lifetime, food_creation_frame, food_lifetime_history
    global last_food_id, total_food_eaten, start_real_time, frame_timestamps
    global flocking_candidates, flocking_confirmed

    # Track real time
    current_real_time = time.time()
    if start_real_time is None:
        start_real_time = current_real_time
    frame_timestamps.append(current_real_time - start_real_time)

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
    if flocking_confirmed and stabilized_frame is not None and not food_released:
        print(f"Food released at frame {i}")
        food_released = True
        food_release_frame = i  # Track the frame food appears
        # Initialize the creation time for all food sources when they are first released
        food_creation_frame = np.full(num_food, i)

    # Handle food behavior only after release
    if food_released and len(food_positions) > 0:
        # Update lifetime for active food sources
        food_lifetime[food_active] += 1

        active_food_positions = food_positions[food_active]

        if len(active_food_positions) > 0:
            tree_food = cKDTree(active_food_positions, boxsize=[L, L])
            d_food, idx_food = tree_food.query(pos, k=1)
            f_pos = active_food_positions[idx_food]
            delta_food = (f_pos - pos + L / 2) % L - L / 2
            norms = np.linalg.norm(delta_food, axis=1)
            unit_food = delta_food / norms[:, None]
            food_vec = unit_food[:, 0] + 1j * unit_food[:, 1]
            S_total += food_attraction_strength * food_vec

            global_indices = np.flatnonzero(food_active)[idx_food]
            eaten = np.unique(global_indices[d_food < eat_radius])
            
            if len(eaten) > 0:
                for food_idx in eaten:
                    # Record lifetime of this food instance (in frames)
                    current_lifetime = food_lifetime[food_idx]
                    food_lifetime_history.append(current_lifetime)
                    
                    # Increment total food eaten counter
                    total_food_eaten += 1
                    
                    # Calculate frames from creation to consumption
                    creation_frame = food_creation_frame[food_idx]
                    consumption_frame = i
                    frame_to_consumption.append(consumption_frame - creation_frame)
                    
                    # Reset lifetime counter for this food
                    food_lifetime[food_idx] = 0
                    
                    # Remember last food consumed
                    last_food_id = food_idx
                    
                    # Debug print
                    print(f"Food at position {food_idx} consumed at frame {i}. Lifetime: {current_lifetime} frames. Total eaten: {total_food_eaten}")

            food_active[eaten] = False
            food_timers[eaten] = 0
    
    # Always process food respawning, regardless of active state
    food_timers[~food_active] += 1
    respawned = (food_timers >= respawn_delay) & (~food_active)

    if np.any(respawned):
        food_active[respawned] = True
        food_timers[respawned] = 0
        # Record creation time for respawned food (in frames)
        food_creation_frame[respawned] = i

    # Update orientations with noise
    orient = np.angle(S_total) + eta * np.random.uniform(-np.pi, np.pi, size=N)

    # Update positions
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0 * speed
    pos[:, 1] += sin * v0 * speed
    pos %= L

    # Update plots
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, orient)

    if food_released:
        sc.set_offsets(food_positions[food_active])
        
        if last_food_id is not None:
            food_text.set_text(f'Total eaten: {total_food_eaten}\n')
    else:
        sc.set_offsets(np.empty((0, 2)))  

    if i == num_frames - 1:
        plt.close(fig)

    # Display frame number and times
    simulation_time = i * 0.01  # Original simulation time units
    elapsed_real_time = frame_timestamps[-1] if frame_timestamps else 0
    time_text.set_text(f'Frame: {i} | Sim Time: {simulation_time:.2f}')
    real_time_text.set_text(f'Real Time: {elapsed_real_time:.2f}s')

    mean_heading = np.mean(np.exp(1j * orient))
    cohesion = np.abs(mean_heading)
    cohesion_history.append(cohesion)
    if food_released:
        post_food_cohesion.append(cohesion)

    if len(cohesion_history) > stabilization_window:
        recent = cohesion_history[-stabilization_window:]
        if max(recent) - min(recent) < stabilization_threshold and stabilized_frame is None:
            stabilized_frame = i
            print(f"Group cohesion stabilized at frame {i} with cohesion ≈ {cohesion:.3f}")
    
    neighbor_graph = sparse.coo_matrix((np.ones_like(dist.data), (dist.row, dist.col)), shape=(N, N))
    n_components, labels = connected_components(neighbor_graph, directed=False)
    flock_history.append(n_components)

    if food_released:
        post_food_flock_counts.append(n_components)

    # Improved flocking detection with confirmation window
    if n_components == 1:
        flocking_candidates.append(i)
    
    # Remove old candidates outside the confirmation window
    flocking_candidates = [frame for frame in flocking_candidates if frame >= i - flocking_confirmation_window + 1]
    
    # Check if we have sustained flocking for the required window
    if len(flocking_candidates) >= flocking_confirmation_window and not flocking_confirmed:
        # Verify that all frames in the window show single flock
        expected_frames = list(range(i - flocking_confirmation_window + 1, i + 1))
        if flocking_candidates == expected_frames:
            flocking_confirmed = True
            global flocking_frame
            flocking_frame = i - flocking_confirmation_window + 1  # First frame of sustained flocking
            print(f"All boids formed one stable flock at frame {flocking_frame} (sustained for {flocking_confirmation_window} frames)")

    return qv, sc, time_text, food_text, real_time_text

num_frames = 2000
anim = FuncAnimation(fig, animate, frames=num_frames, interval=10, blit=True, repeat=False)
plt.show()

# Plot original metrics with frame-based x-axis
plt.figure()
if food_release_frame is not None:
    frames = np.arange(food_release_frame)
    plt.plot(frames, cohesion_history[:food_release_frame])
else:
    frames = np.arange(len(cohesion_history))
    plt.plot(frames, cohesion_history)
plt.xlabel("Frame")
plt.ylabel("Cohesion")
plt.title("Group Cohesion (Before Food)")
plt.grid(True)
plt.show()

if food_release_frame is not None and len(post_food_cohesion) > 0:
    plt.figure()
    frames_post = np.arange(len(post_food_cohesion))  # Frames after food appears
    plt.plot(frames_post, post_food_cohesion)
    plt.xlabel("Frames After Food Appears")
    plt.ylabel("Cohesion")
    plt.title("Group Cohesion (After Food)")
    plt.grid(True)
    plt.show()

plt.figure()
if food_release_frame is not None:
    frames = np.arange(food_release_frame)
    plt.plot(frames, flock_history[:food_release_frame])
else:
    frames = np.arange(len(flock_history))
    plt.plot(frames, flock_history)
plt.xlabel("Frame")
plt.ylabel("Number of Flocks")
plt.title("Flock Count (Before Food)")
plt.grid(True)
plt.show()

if food_release_frame is not None and len(post_food_flock_counts) > 0:
    plt.figure()
    frames_post = np.arange(len(post_food_flock_counts))  # Frames after food appears
    plt.plot(frames_post, post_food_flock_counts)
    plt.xlabel("Frames After Food Appears")
    plt.ylabel("Number of Flocks")
    plt.title("Flock Count (After Food)")
    plt.grid(True)
    plt.show()

# Remaining visualizations for food metrics
if food_release_frame is not None and len(food_lifetime_history) > 0:
    # Food lifetime distribution
    plt.figure(figsize=(8, 6))
    plt.hist(food_lifetime_history, bins=10, color='green', alpha=0.7)
    plt.xlabel("Lifetime (frames)")
    plt.ylabel("Frequency")
    plt.title(f"Food Lifetime Distribution (Mean: {np.mean(food_lifetime_history):.1f} frames)")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Frames from creation to consumption
    if len(frame_to_consumption) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(frame_to_consumption, bins=10, color='purple', alpha=0.7)
        plt.xlabel("Frames from Creation to Consumption")
        plt.ylabel("Frequency")
        plt.title(f"Food Discovery Speed (Mean: {np.mean(frame_to_consumption):.1f} frames)")
        plt.grid(True, alpha=0.3)
        plt.show()

# Summary statistics (print to console)
if food_release_frame is not None and len(food_lifetime_history) > 0:
    print("\n===== FOOD LIFETIME METRICS SUMMARY =====")
    print(f"TOTAL FOOD EATEN IN EXPERIMENT: {total_food_eaten}")
    print(f"Total number of food consumptions: {len(food_lifetime_history)}")
    print(f"Average food lifetime: {np.mean(food_lifetime_history):.2f} frames")
    print(f"Max food lifetime: {np.max(food_lifetime_history)} frames")
    print(f"Min food lifetime: {np.min(food_lifetime_history)} frames")
    
    if len(frame_to_consumption) > 0:
        print(f"Average frames from creation to consumption: {np.mean(frame_to_consumption):.2f} frames")
else:
    print(f"\n===== EXPERIMENT SUMMARY =====")
    print(f"TOTAL FOOD EATEN IN EXPERIMENT: {total_food_eaten}")

# Print timing comparison
if frame_timestamps:
    total_real_time = frame_timestamps[-1]
    total_sim_time = (len(frame_timestamps) - 1) * 0.01
    print(f"\n===== TIMING COMPARISON =====")
    print(f"Total frames processed: {len(frame_timestamps)}")
    print(f"Total real time: {total_real_time:.2f} seconds")
    print(f"Total simulation time units: {total_sim_time:.2f}")
    print(f"Real time per frame: {total_real_time/len(frame_timestamps)*1000:.2f} ms")

# Save raw data to file
# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Format filename according to specifications
filename = f"experiment-food_amount_{num_food}-urge_{food_attraction_strength}.txt"
filepath = os.path.join(output_dir, filename)

# Write all raw data to file
with open(filepath, 'w') as f:
    f.write("===== EXPERIMENT RAW DATA =====\n")
    f.write(f"Experiment Parameters:\n")
    f.write(f"  Number of particles (N): {N}\n")
    f.write(f"  System size (L): {L}\n")
    f.write(f"  Particle density (rho): {rho}\n")
    f.write(f"  Number of food sources: {num_food}\n")
    f.write(f"  Food attraction strength: {food_attraction_strength}\n")
    f.write(f"  Simulation frames: {num_frames}\n")
    f.write(f"  Random seed: 123\n")
    f.write(f"  Speed: {speed}\n")
    f.write(f"  Alignment weight: {w_align}\n")
    f.write(f"  Cohesion weight: {w_cohesion}\n")
    f.write(f"  Separation weight: {w_separation}\n")
    f.write(f"  Interaction radius: {r0}\n")
    f.write(f"  Noise parameter: {eta}\n")
    f.write(f"  Eat radius: {eat_radius}\n")
    f.write(f"  Respawn delay: {respawn_delay} frames\n")
    f.write(f"  Flocking confirmation window: {flocking_confirmation_window} frames\n")
    f.write("\n")
    
    # Timing information
    if frame_timestamps:
        total_real_time = frame_timestamps[-1]
        total_sim_time = (len(frame_timestamps) - 1) * 0.01
        f.write("===== TIMING INFORMATION =====\n")
        f.write(f"Total frames processed: {len(frame_timestamps)}\n")
        f.write(f"Total real time: {total_real_time:.2f} seconds\n")
        f.write(f"Total simulation time units: {total_sim_time:.2f}\n")
        f.write(f"Real time per frame: {total_real_time/len(frame_timestamps)*1000:.2f} ms\n")
        f.write("\n")
    
    f.write("===== SIMULATION EVENTS =====\n")
    if flocking_confirmed:
        f.write(f"Stable flocking achieved at frame: {flocking_frame}\n")
    else:
        f.write("Stable flocking not achieved\n")
    
    if stabilized_frame is not None:
        f.write(f"Cohesion stabilized at frame: {stabilized_frame}\n")
    else:
        f.write("Cohesion not stabilized\n")
    
    if food_release_frame is not None:
        f.write(f"Food released at frame: {food_release_frame}\n")
    else:
        f.write("Food not released\n")
    f.write("\n")
    
    f.write("===== FOOD METRICS SUMMARY =====\n")
    f.write(f"Total food eaten: {total_food_eaten}\n")
    if len(food_lifetime_history) > 0:
        f.write(f"Number of food consumptions: {len(food_lifetime_history)}\n")
        f.write(f"Average food lifetime: {np.mean(food_lifetime_history):.2f} frames\n")
        f.write(f"Max food lifetime: {np.max(food_lifetime_history)} frames\n")
        f.write(f"Min food lifetime: {np.min(food_lifetime_history)} frames\n")
        f.write(f"Food lifetime std dev: {np.std(food_lifetime_history):.2f} frames\n")
    
    if len(frame_to_consumption) > 0:
        f.write(f"Average frames from creation to consumption: {np.mean(frame_to_consumption):.2f} frames\n")
        f.write(f"Max frames to consumption: {np.max(frame_to_consumption)} frames\n")
        f.write(f"Min frames to consumption: {np.min(frame_to_consumption)} frames\n")
        f.write(f"Frames to consumption std dev: {np.std(frame_to_consumption):.2f} frames\n")
    f.write("\n")
    
    f.write("===== RAW DATA ARRAYS =====\n")
    
    # Cohesion history (all frames)
    f.write("Cohesion History (all frames):\n")
    f.write("Frame,Cohesion\n")
    for i, cohesion in enumerate(cohesion_history):
        f.write(f"{i},{cohesion:.6f}\n")
    f.write("\n")
    
    # Cohesion history before food
    if food_release_frame is not None:
        f.write("Cohesion History (before food release):\n")
        f.write("Frame,Cohesion\n")
        for i, cohesion in enumerate(cohesion_history[:food_release_frame]):
            f.write(f"{i},{cohesion:.6f}\n")
        f.write("\n")
        
        # Cohesion history after food
        if len(post_food_cohesion) > 0:
            f.write("Cohesion History (after food release):\n")
            f.write("Frame_After_Food,Cohesion\n")
            for i, cohesion in enumerate(post_food_cohesion):
                f.write(f"{i},{cohesion:.6f}\n")
            f.write("\n")
    
    # Flock count history (all frames)
    f.write("Flock Count History (all frames):\n")
    f.write("Frame,Flock_Count\n")
    for i, count in enumerate(flock_history):
        f.write(f"{i},{count}\n")
    f.write("\n")
    
    # Flock count before food
    if food_release_frame is not None:
        f.write("Flock Count History (before food release):\n")
        f.write("Frame,Flock_Count\n")
        for i, count in enumerate(flock_history[:food_release_frame]):
            f.write(f"{i},{count}\n")
        f.write("\n")
        
        # Flock count after food
        if len(post_food_flock_counts) > 0:
            f.write("Flock Count History (after food release):\n")
            f.write("Frame_After_Food,Flock_Count\n")
            for i, count in enumerate(post_food_flock_counts):
                f.write(f"{i},{count}\n")
            f.write("\n")
    
    # Food lifetime data
    if len(food_lifetime_history) > 0:
        f.write("Food Lifetime Data:\n")
        f.write("Consumption_Number,Lifetime_Frames\n")
        for i, lifetime in enumerate(food_lifetime_history):
            f.write(f"{i+1},{lifetime}\n")
        f.write("\n")
    
    # Frames to consumption data
    if len(frame_to_consumption) > 0:
        f.write("Frames from Creation to Consumption:\n")
        f.write("Consumption_Number,Frames_To_Consumption\n")
        for i, frames_val in enumerate(frame_to_consumption):
            f.write(f"{i+1},{frames_val}\n")
        f.write("\n")
    
    # Real time data
    if frame_timestamps:
        f.write("Frame Timestamps (Real Time):\n")
        f.write("Frame,Real_Time_Seconds\n")
        for i, timestamp in enumerate(frame_timestamps):
            f.write(f"{i},{timestamp:.6f}\n")
        f.write("\n")
    
    # Food positions
    f.write("Food Positions:\n")
    f.write("Food_ID,X_Position,Y_Position\n")
    for i, pos in enumerate(food_positions):
        f.write(f"{i},{pos[0]:.6f},{pos[1]:.6f}\n")
    f.write("\n")

print(f"\nRaw data saved to: {filepath}")