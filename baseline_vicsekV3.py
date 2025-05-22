import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.csgraph import connected_components
import time

np.random.seed(123) 

# Timing variables
start_time = time.time()  # Record real-world start time
frame_to_mtu_conversion = 0.01   # Model Time Unit conversion (100 frames = 1 MTU)

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
print(f"Number of boids: {N}")

# Particle movement parameters 
w_align = 1.2      # Weight for Vicsek alignment
w_cohesion = 0.6   # Weight for Boids cohesion
w_separation = 0.15  # Weight for Boids separation
separation_radius = 0.5  # Personal space radius
r0 = 2 # Interaction radius
deltat = 1.0  # Time step (unused)
factor = 0.5
v0 = r0/deltat*factor  # Base speed
eta_base = 0.05  # Base noise parameter

# Food parameters
num_food = 4  # number of food sources (fixed positions)
eat_radius = 1.0  # Radius at which food is consumed
food_attraction_base = 2.0  # Base attraction strength for food

# Static urge parameter (instead of dynamic urge)
static_urge = 0.5  # Fixed urge level between min (0.2) and max (1.0)
urge_history = []  # Track urge over time (will be constant now)

# Enhanced food lifetime metrics
food_lifetime = np.zeros(num_food, dtype=int)  # Track frames each food exists
food_creation_time = np.zeros(num_food, dtype=int)  # When each food was created/respawned
food_consumption_count = np.zeros(num_food, dtype=int)  # How many times each food spot was consumed
food_lifetime_history = []  # Track the lifetime of each consumed food
food_consumption_times = []  # Track when food was consumed (frame number)
food_consumption_positions = []  # Track which food position was consumed
food_total_frames_active = np.zeros(num_food, dtype=int)  # Total frames each position was active
food_availability_ratio = []  # Ratio of frames food was available
last_food_id = None  # Track the ID of the last food consumed

# Initialize particles
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)

# Initialize static urge levels for each boid
urge = np.full(N, static_urge)

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

# If you want a square grid (2x2, 3x3, etc.)
grid_side = int(np.ceil(np.sqrt(num_food)))
food_positions = create_evenly_spaced_food(grid_side, grid_side, L)[:num_food]

# Track which food sources are "active" (True = available)
food_active = np.ones(num_food, dtype=bool)
respawn_delay = 10  # frames (0.1 second at 10 ms/frame)
food_timers = np.zeros(num_food, dtype=int)
food_released = False # only release food spawn after flock has stabilized

# New: For calculating statistical metrics
food_encounter_counts = np.zeros(num_food, dtype=int)  # Track how many boids get close to each food
mean_time_to_consumption = []  # Average time from spawn to consumption

# Setup visualization
fig, ax = plt.subplots(figsize=(8, 8))
cos0, sin0 = np.cos(orient), np.sin(orient)
qv = ax.quiver(pos[:, 0], pos[:, 1], cos0, sin0, orient, clim=[-np.pi, np.pi])
sc = ax.scatter(food_positions[:, 0], food_positions[:, 1], marker='o', s=100, color='red')

# Removed colored dots for urge and colorbar

# Text elements for displays
frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')
elapsed_text = ax.text(0.7, 0.95, '', transform=ax.transAxes, fontsize=10, color='darkgreen')
food_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=10, color='blue')
flock_info_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10, color='purple')

def animate(i):
    global orient, pos, food_active, stabilized_frame, food_released, food_release_frame
    global food_lifetime, food_creation_time, food_consumption_count, food_lifetime_history
    global food_consumption_times, food_consumption_positions, last_food_id
    global food_total_frames_active, food_availability_ratio
    global urge  # Keep static urge reference
    global flocking_frame  # Make sure this is defined
    
    # Track real-world elapsed time
    real_elapsed = time.time() - start_time
    # Model Time Units calculation
    mtu = i * frame_to_mtu_conversion

    # No longer updating urge levels - they remain static
    
    # Neighbor interactions
    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type='coo_matrix')

    # Precompute neighbor contributions
    orient_neighbors = orient[dist.col]
    pos_neighbors = pos[dist.col]
    pos_self = pos[dist.row]

    delta_pos = (pos_neighbors - pos_self + L / 2) % L - L / 2

    # ---------- ALIGNMENT (Vicsek model) ----------
    alignment_vec = np.exp(1j * orient_neighbors)
    alignment_matrix = sparse.coo_matrix((alignment_vec, (dist.row, dist.col)), shape=(N, N))
    alignment_sum = np.squeeze(np.asarray(alignment_matrix.tocsr().sum(axis=1)))

    # ---------- COHESION (Boids model) ----------
    cohesion_matrix = sparse.coo_matrix((delta_pos[:, 0] + 1j * delta_pos[:, 1], (dist.row, dist.col)), shape=(N, N))
    cohesion_vec = np.squeeze(np.asarray(cohesion_matrix.tocsr().sum(axis=1)))

    # Count neighbors for each boid for density calculations
    neighbor_counts = np.bincount(dist.row, minlength=N)
    
    # ---------- SEPARATION (Boids model) ----------
    separation_force = -delta_pos / (np.linalg.norm(delta_pos, axis=1, keepdims=True) + 1e-8)
    mask = np.linalg.norm(delta_pos, axis=1) < separation_radius
    separation_force = separation_force[mask]
    sep_rows = dist.row[mask]
    separation_complex = separation_force[:, 0] + 1j * separation_force[:, 1]
    separation_matrix = sparse.coo_matrix((separation_complex, (sep_rows, sep_rows)), shape=(N, N))
    separation_vec = np.squeeze(np.asarray(separation_matrix.tocsr().sum(axis=1)))

    # Initialize total steering force with flocking behaviors
    S_total = (
        w_align * alignment_sum +
        w_cohesion * cohesion_vec +
        w_separation * separation_vec
    )

    # Initialize flocking_frame before using it
    if 'flocking_frame' not in globals():
        global flocking_frame
        flocking_frame = None
    
    # Release food only after both cohesion and flocking conditions are met
    if flocking_frame is not None and stabilized_frame is not None and not food_released:
        print(f"Food released at frame {i} (MTU: {mtu:.2f}, Real time: {real_elapsed:.2f}s)")
        food_released = True
        food_release_frame = i  # Track the frame food appears
        # Initialize the creation time for all food sources when they are first released
        food_creation_time = np.full(num_food, i)

    # Handle food behavior only after release
    food_vec = np.zeros(N, dtype=complex)  # Initialize empty food vector for all boids
    boids_eating = []  # Track which boids eat food in this frame
    
    if food_released and len(food_positions) > 0:
        # Update lifetime for active food sources
        food_lifetime[food_active] += 1
        food_total_frames_active[food_active] += 1
        
        # Update food availability ratio
        if i > food_release_frame:
            availability = np.sum(food_active) / num_food
            food_availability_ratio.append(availability)
        
        active_food_positions = food_positions[food_active]

        if len(active_food_positions) > 0:
            tree_food = cKDTree(active_food_positions, boxsize=[L, L])
            d_food, idx_food = tree_food.query(pos, k=1)
            f_pos = active_food_positions[idx_food]
            delta_food = (f_pos - pos + L / 2) % L - L / 2
            norms = np.linalg.norm(delta_food, axis=1)
            
            # Apply fixed food attraction strength instead of sigmoid scaling
            food_attraction_strength = food_attraction_base
            
            # Calculate unit vectors for direction to food, handling zero distances
            unit_food = np.zeros_like(delta_food)
            nonzero_mask = norms > 0
            unit_food[nonzero_mask] = delta_food[nonzero_mask] / norms[nonzero_mask, None]
            
            # Create complex vector for food attraction
            food_vec = unit_food[:, 0] + 1j * unit_food[:, 1]
            
            # Add food attraction to total steering force with fixed strength
            S_total += food_attraction_strength * food_vec

            # Count boids that are "interested" in each food source (within 2x eat_radius)
            for food_idx in range(len(active_food_positions)):
                food_interest_count = np.sum(d_food < (2 * eat_radius))
                if food_interest_count > 0:
                    global_idx = np.flatnonzero(food_active)[food_idx]
                    food_encounter_counts[global_idx] += food_interest_count

            # Find which boids are close enough to eat food
            eating_boids = d_food < eat_radius
            if np.any(eating_boids):
                boids_eating = np.where(eating_boids)[0]
                # No longer modifying urge for boids that eat - urge remains static
            
            global_indices = np.flatnonzero(food_active)[idx_food]
            eaten = np.unique(global_indices[eating_boids])
            
            if len(eaten) > 0:
                for food_idx in eaten:
                    # Record lifetime of this food instance
                    current_lifetime = food_lifetime[food_idx]
                    food_lifetime_history.append(current_lifetime)
                    
                    # Record when and which food was consumed
                    food_consumption_times.append(i)
                    food_consumption_positions.append(food_idx)
                    
                    # Increment consumption count for this position
                    food_consumption_count[food_idx] += 1
                    
                    # Calculate time from creation to consumption
                    if len(food_consumption_times) > 1:
                        creation_time = food_creation_time[food_idx]
                        consumption_time = i
                        mean_time_to_consumption.append(consumption_time - creation_time)
                    
                    # Reset lifetime counter for this food
                    food_lifetime[food_idx] = 0
                    
                    # Remember last food consumed
                    last_food_id = food_idx
                    
                    # Debug print
                    print(f"Food at position {food_idx} consumed at frame {i}. Lifetime: {current_lifetime} frames, MTU: {current_lifetime * frame_to_mtu_conversion:.2f}")
                    
                # Deactivate eaten food
                food_active[eaten] = False
                food_timers[eaten] = 0
    
    # Always process food respawning, regardless of active state
    food_timers[~food_active] += 1
    respawned = (food_timers >= respawn_delay) & (~food_active)
    
    if np.any(respawned):
        food_active[respawned] = True
        food_timers[respawned] = 0
        # Record creation time for respawned food
        food_creation_time[respawned] = i

    # Use fixed noise level - not dependent on urge anymore
    individual_noise = eta_base * np.random.uniform(-np.pi, np.pi, size=N)
    
    # Update orientations with fixed noise
    magnitudes = np.abs(S_total)
    zero_magnitude = magnitudes < 1e-10
    
    if np.any(zero_magnitude):
        # For boids with zero steering force, maintain current orientation with noise
        orient[~zero_magnitude] = np.angle(S_total[~zero_magnitude]) + individual_noise[~zero_magnitude]
        # Add just noise to current orientation for those with zero steering
        orient[zero_magnitude] += individual_noise[zero_magnitude]
    else:
        orient = np.angle(S_total) + individual_noise

    # Update positions
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0 * speed
    pos[:, 1] += sin * v0 * speed
    pos %= L

    # Track urge history (will be a constant line now)
    urge_history.append(np.mean(urge))

    # Update plots
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, orient)

    if food_released:
        sc.set_offsets(food_positions[food_active])
        
        # Display food metrics on screen
        if last_food_id is not None:
            food_text.set_text(f'Last food consumed: {last_food_id}, Times: {food_consumption_count[last_food_id]}, '
                             f'Avg Lifetime: {np.mean(food_lifetime_history) if food_lifetime_history else 0:.1f} frames')
    else:
        sc.set_offsets(np.empty((0, 2)))  

    if i == num_frames - 1:
        plt.close(fig)

    # Update display texts
    frame_text.set_text(f'Frame: {i}')
    elapsed_text.set_text(f'Real time: {real_elapsed:.2f}s')

    # Calculate cohesion metric
    mean_heading = np.mean(np.exp(1j * orient))
    cohesion = np.abs(mean_heading)
    cohesion_history.append(cohesion)
    if food_released:
        post_food_cohesion.append(cohesion)

    # Check for stabilization
    if len(cohesion_history) > stabilization_window:
        recent = cohesion_history[-stabilization_window:]
        if max(recent) - min(recent) < stabilization_threshold and stabilized_frame is None:
            stabilized_frame = i
            print(f"Group cohesion stabilized at frame {i} (MTU: {mtu:.2f}, Real time: {real_elapsed:.2f}s) with cohesion ≈ {cohesion:.3f}")
    
    # Count separate flocks
    neighbor_graph = sparse.coo_matrix((np.ones_like(dist.data), (dist.row, dist.col)), shape=(N, N))
    n_components, labels = connected_components(neighbor_graph, directed=False)
    flock_history.append(n_components)

    # Update flock count display
    flock_info_text.set_text(f'Flocks: {n_components}, Cohesion: {cohesion:.3f}')

    if food_released:
        post_food_flock_counts.append(n_components)

    if n_components == 1 and flocking_frame is None:
        flocking_frame = i
        print(f"All boids formed one flock at frame {i} (MTU: {mtu:.2f}, Real time: {real_elapsed:.2f}s)")

    return qv, sc, frame_text, food_text, elapsed_text, flock_info_text

num_frames = 2000
anim = FuncAnimation(fig, animate, frames=num_frames, interval=10, blit=True, repeat=False)
plt.show()

# Plot original metrics
plt.figure(figsize=(12, 8))

# Plot 1: Group Cohesion Before Food
plt.subplot(2, 2, 1)
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

# Plot 2: Group Cohesion After Food
plt.subplot(2, 2, 2)
if food_release_frame is not None and len(post_food_cohesion) > 0:
    frames_post = np.arange(len(post_food_cohesion))  # Frames after food appears
    plt.plot(frames_post, post_food_cohesion)
    plt.xlabel("Frames After Food Release")
    plt.ylabel("Cohesion")
    plt.title("Group Cohesion (After Food)")
    plt.grid(True)

# Plot 3: Flock Count Before Food
plt.subplot(2, 2, 3)
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

# Plot 4: Flock Count After Food
plt.subplot(2, 2, 4)
if food_release_frame is not None and len(post_food_flock_counts) > 0:
    frames_post = np.arange(len(post_food_flock_counts))  # Frames after food appears
    plt.plot(frames_post, post_food_flock_counts)
    plt.xlabel("Frames After Food Release")
    plt.ylabel("Number of Flocks")
    plt.title("Flock Count (After Food)")
    plt.grid(True)

plt.tight_layout()
plt.show()

# Remove urge plot that's no longer needed
# Plot food metrics visualizations
if food_release_frame is not None and len(food_lifetime_history) > 0:
    # 1. Food consumption frequency and lifetime distribution
    plt.figure(figsize=(12, 6))
    
    # Create a histogram of food consumption frequencies
    plt.subplot(1, 2, 1)
    plt.bar(range(num_food), food_consumption_count)
    plt.xlabel("Food ID")
    plt.ylabel("Number of Consumptions")
    plt.title("Food Consumption Frequency")
    plt.xticks(range(num_food))
    plt.grid(True, alpha=0.3)
    
    # Create a histogram of food lifetimes
    plt.subplot(1, 2, 2)
    plt.hist(food_lifetime_history, bins=10, color='green', alpha=0.7)
    plt.xlabel("Lifetime (frames)")
    plt.ylabel("Frequency")
    plt.title(f"Food Lifetime Distribution (Mean: {np.mean(food_lifetime_history):.1f} frames)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Food consumption timeline
    if len(food_consumption_times) > 0:
        plt.figure(figsize=(10, 6))
        # Convert to frames after food release
        consumption_frames = [t - food_release_frame for t in food_consumption_times]
        food_positions = food_consumption_positions
        
        plt.scatter(consumption_frames, food_positions, marker='o', s=50, alpha=0.7)
        plt.xlabel("Frames After Food Release")
        plt.ylabel("Food Position ID")
        plt.title("Food Consumption Timeline")
        plt.grid(True)
        plt.yticks(range(num_food))
        
        # Add trend line if multiple consumptions
        if len(consumption_frames) > 1:
            z = np.polyfit(consumption_frames, food_positions, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(consumption_frames), max(consumption_frames), 100)
            plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        plt.show()
    
    # 3. Food availability ratio over time
    if len(food_availability_ratio) > 0:
        plt.figure(figsize=(8, 5))
        frames_availability = np.arange(len(food_availability_ratio))
        plt.plot(frames_availability, food_availability_ratio, 'b-')
        plt.xlabel("Frames After Food Release")
        plt.ylabel("Food Availability Ratio")
        plt.title("Proportion of Food Available Over Time")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.show()
    
    # 4. Time from creation to consumption
    if len(mean_time_to_consumption) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(mean_time_to_consumption, bins=10, color='purple', alpha=0.7)
        plt.xlabel("Frames from Creation to Consumption")
        plt.ylabel("Frequency")
        plt.title(f"Food Discovery Speed (Mean: {np.mean(mean_time_to_consumption):.1f} frames)")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Total activity time for each food position
        plt.figure(figsize=(8, 5))
        plt.bar(range(num_food), food_total_frames_active)
        plt.xlabel("Food ID")
        plt.ylabel("Total Frames Active")
        plt.title("Total Activity Time per Food Position")
        plt.xticks(range(num_food))
        plt.grid(True, alpha=0.3)
        plt.show()

# Summary statistics (print to console)
if food_release_frame is not None and len(food_lifetime_history) > 0:
    # Calculate simulation end time to get total run duration
    end_time = time.time()
    total_run_time = end_time - start_time
    total_frames = num_frames
    
    print("\n===== FOOD LIFETIME METRICS SUMMARY =====")
    print(f"Simulation run time: {total_run_time:.2f} seconds wall clock time")
    print(f"Total frames: {total_frames}")
    print(f"Average frame rate: {total_frames/total_run_time:.2f} frames per second")
    print(f"Food released at frame: {food_release_frame}")
    
    print(f"\nTotal number of food consumptions: {len(food_lifetime_history)}")
    print(f"Average food lifetime: {np.mean(food_lifetime_history):.2f} frames")
    print(f"Max food lifetime: {np.max(food_lifetime_history)} frames")
    print(f"Min food lifetime: {np.min(food_lifetime_history)} frames")
    
    if len(mean_time_to_consumption) > 0:
        print(f"Average time from creation to consumption: {np.mean(mean_time_to_consumption):.2f} frames")
    
    most_consumed = np.argmax(food_consumption_count)
    print(f"Most consumed food position: ID {most_consumed} with {food_consumption_count[most_consumed]} consumptions")
    
    print("\nFood positions consumption statistics:")
    for i in range(num_food):
        print(f"  Food ID {i}: {food_consumption_count[i]} consumptions, {food_total_frames_active[i]} frames active")
    
    print("=========================================\n")