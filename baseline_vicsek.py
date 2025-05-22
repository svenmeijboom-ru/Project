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
speed = 0.5
L = 32.0  # System size
rho = 3.0  # Particle density
N = int(rho*30)  # Number of particles
print(N)

# Particle movement parameters
w_align = 0.3      # Weight for Vicsek alignment
w_cohesion = 0.1   # Weight for Boids cohesion
w_separation = 0.5  # Weight for Boids separation
separation_radius = 0.3
r0 = 1 # Interaction radius
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


# Create fixed food positions (uniformly spaced in a 2D grid)
# grid_side = int(np.ceil(np.sqrt(num_food)))
# x_coords = np.linspace(0.2 * L, 0.8 * L, grid_side)
# y_coords = np.linspace(0.2 * L, 0.8 * L, grid_side)
# xx, yy = np.meshgrid(x_coords, y_coords)
# food_positions = np.vstack([xx.ravel(), yy.ravel()]).T[:num_food]

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
respawn_delay = 100  # frames (0.1 second at 10 ms/frame)
food_timers = np.zeros(num_food, dtype=int)
food_released = False # only release food spawn after flock has stabilized

# New: For calculating statistical metrics
food_encounter_counts = np.zeros(num_food, dtype=int)  # Track how many boids get close to each food
mean_time_to_consumption = []  # Average time from spawn to consumption


fig, ax = plt.subplots(figsize=(6, 6))
cos0, sin0 = np.cos(orient), np.sin(orient)
qv = ax.quiver(pos[:, 0], pos[:, 1], cos0, sin0, orient, clim=[-np.pi, np.pi])
sc = ax.scatter(food_positions[:, 0], food_positions[:, 1], marker='o', s=100, color='red')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')
food_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=10, color='blue')

def animate(i):
    global orient, pos, food_active, stabilized_frame, food_released, food_release_frame
    global food_lifetime, food_creation_time, food_consumption_count, food_lifetime_history
    global food_consumption_times, food_consumption_positions, last_food_id
    global food_total_frames_active, food_availability_ratio

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
        # Initialize the creation time for all food sources when they are first released
        food_creation_time = np.full(num_food, i)

    # Handle food behavior only after release
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
            unit_food = delta_food / norms[:, None]
            food_vec = unit_food[:, 0] + 1j * unit_food[:, 1]
            S_total += food_attraction_strength * food_vec

            # Count boids that are "interested" in each food source (within 2x eat_radius)
            for food_idx in range(len(active_food_positions)):
                food_interest_count = np.sum(d_food < (2 * eat_radius))
                if food_interest_count > 0:
                    global_idx = np.flatnonzero(food_active)[food_idx]
                    food_encounter_counts[global_idx] += food_interest_count

            global_indices = np.flatnonzero(food_active)[idx_food]
            eaten = np.unique(global_indices[d_food < eat_radius])
            
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
                    print(f"Food at position {food_idx} consumed at frame {i}. Lifetime: {current_lifetime} frames")


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

    # Update orientations with noise
    orient = np.angle(S_total) + eta * np.random.uniform(-np.pi, np.pi, size=N)

    # Update positions
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0 * speed
    pos[:, 1] += sin * v0 * speed
    pos %= L

    # Handle eating
    # if food_released and len(food_positions) > 0:
    #     if np.any(food_active):
    #         global_indices = np.flatnonzero(food_active)[idx_food]
    #         eaten = np.unique(global_indices[d_food < eat_radius])
    #         food_active[eaten] = False
    #         food_timers[eaten] = 0

    #     food_timers[~food_active] += 1
    #     respawned = (food_timers >= respawn_delay) & (~food_active)
    #     food_active[respawned] = True
    #     food_timers[respawned] = 0

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

    return qv, sc, time_text, food_text

num_frames = 2000
anim = FuncAnimation(fig, animate, frames=num_frames, interval=10, blit=True, repeat=False)
plt.show()

# Plot original metrics
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

# New visualizations for food metrics
if food_release_frame is not None and len(food_lifetime_history) > 0:
    # 1. Food consumption frequency
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
        consumption_times_sec = [(t - food_release_frame) * 0.01 for t in food_consumption_times]
        food_positions = food_consumption_positions
        
        plt.scatter(consumption_times_sec, food_positions, marker='o', s=50, alpha=0.7)
        plt.xlabel("Time After Food Release (s)")
        plt.ylabel("Food Position ID")
        plt.title("Food Consumption Timeline")
        plt.grid(True)
        plt.yticks(range(num_food))
        
        # Add trend line if multiple consumptions
        if len(consumption_times_sec) > 1:
            z = np.polyfit(consumption_times_sec, food_positions, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(consumption_times_sec), max(consumption_times_sec), 100)
            plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        plt.show()
    
    # 3. Food availability ratio over time
    if len(food_availability_ratio) > 0:
        plt.figure(figsize=(8, 5))
        t_availability = np.arange(len(food_availability_ratio)) * 0.01
        plt.plot(t_availability, food_availability_ratio, 'b-')
        plt.xlabel("Time After Food Release (s)")
        plt.ylabel("Food Availability Ratio")
        plt.title("Proportion of Food Available Over Time")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.show()
    
    # 4. Time from creation to consumption
    if len(mean_time_to_consumption) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(mean_time_to_consumption, bins=10, color='purple', alpha=0.7)
        plt.xlabel("Time from Creation to Consumption (frames)")
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
    print("\n===== FOOD LIFETIME METRICS SUMMARY =====")
    print(f"Total number of food consumptions: {len(food_lifetime_history)}")
    print(f"Average food lifetime: {np.mean(food_lifetime_history):.2f} frames ({np.mean(food_lifetime_history)*0.01:.3f} seconds)")
    print(f"Max food lifetime: {np.max(food_lifetime_history)} frames ({np.max(food_lifetime_history)*0.01:.3f} seconds)")
    print(f"Min food lifetime: {np.min(food_lifetime_history)} frames ({np.min(food_lifetime_history)*0.01:.3f} seconds)")
    
    if len(mean_time_to_consumption) > 0:
        print(f"Average time from creation to consumption: {np.mean(mean_time_to_consumption):.2f} frames")
    
    most_consumed = np.argmax(food_consumption_count)
    print(f"Most consumed food position: ID {most_consumed} with {food_consumption_count[most_consumed]} consumptions")
    
    print("\nFood positions consumption statistics:")
    for i in range(num_food):
        print(f"  Food ID {i}: {food_consumption_count[i]} consumptions, {food_total_frames_active[i]} frames active")
    