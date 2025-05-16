import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time  # Import time module for tracking
from matplotlib.patches import Circle
from collections import deque  # For storing cohesion history

# Simulation parameters
speed = 0.3
L = 32.0  # System size
rho = 3.0  # Particle density
N = int(rho*100)  # Number of particles

# Particle movement parameters
r0 = 2  # Interaction radius
deltat = 1.0  # Time step (unused)
factor = 0.5
v0 = r0/deltat*factor  # Base speed
eta = 0.1  # Noise parameter

# Food parameters
eat_radius = 0.5  # distance at which food is eaten
food_attraction_strength = 0  # strength of attraction to food
food_respawn_delay = 50  # steps to wait before respawning
food_starting_amount = 1.0 
food_radius = 2.0 
food_consumption_rate = 0.03  # Define the missing food consumption rate parameter

# Manually defined food sources - format: [x, y]
# Each row represents the coordinates of one food source
food_sources = np.array([
    [5.0, 5.0],
    [15.0, 5.0],
    [25.0, 5.0],
    [5.0, 15.0],
    [15.0, 15.0], 
    [25.0, 15.0],
    [5.0, 25.0],
    [15.0, 25.0],
    [25.0, 25.0]
])

num_food = len(food_sources)  # number of food sources

# Initialize particles
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)

# Cohesion metrics tracking
cohesion_history = deque(maxlen=100)  # Store recent cohesion values
clustering_history = deque(maxlen=100)  # Store recent clustering values
polarization_history = deque(maxlen=100)  # Store recent polarization values
cohesion_radius = r0 * 2  # Radius to measure clustering

# Initialize food positions and tracking variables
food_positions = food_sources.copy()  # Use the manually defined positions
food_active = np.ones(num_food, dtype=bool)  # Whether food is currently active
food_respawn_counters = np.zeros(num_food)  # Counter for respawn delay
food_creation_time = np.zeros(num_food)  # When each food was created
food_lifetimes = []  # List to store lifetimes of eaten food
food_amounts = np.ones(num_food) * food_starting_amount  # Current amount of food (1.0 = full, 0.0 = empty)
current_time = 0  # Current simulation time

# Initialize food creation times
for i in range(num_food):
    food_creation_time[i] = current_time

# Stats for analysis
avg_lifetime = 0
min_lifetime = float('inf')
max_lifetime = 0
avg_cohesion = 0
avg_clustering = 0
avg_polarization = 0

# Create a figure with appropriate spacing for stats outside the grid
fig = plt.figure(figsize=(10, 8))
grid_spec = plt.GridSpec(1, 2, width_ratios=[4, 1])  # 4:1 ratio for simulation:stats

# Main simulation axis
ax = fig.add_subplot(grid_spec[0])
ax.set_xlim(0, L)
ax.set_ylim(0, L)

# Stats axis (no frame, used for text only)
stats_ax = fig.add_subplot(grid_spec[1])
stats_ax.axis('off')  # Hide the axis

# Plot particles
cos0, sin0 = np.cos(orient), np.sin(orient)
qv = ax.quiver(pos[:, 0], pos[:, 1], cos0, sin0, orient, clim=[-np.pi, np.pi])

# Initialize food visualization as circles
food_plots = []
for i in range(num_food):
    circle = Circle(food_positions[i], food_radius, alpha=0.3, color='red')
    food_plots.append(ax.add_patch(circle))

# Add text for statistics display in the stats area
stats_text = stats_ax.text(0.05, 0.95, '', transform=stats_ax.transAxes, va='top', fontsize=10)
cohesion_text = stats_ax.text(0.05, 0.65, '', transform=stats_ax.transAxes, va='top', fontsize=10)

plt.tight_layout()

def calculate_cohesion_metrics(positions, orientations, tree, r):
    """Calculate metrics for group cohesion"""
    # 1. Average distance between particles
    # Use the KDTree to find all pairs within a certain distance
    pairs = tree.sparse_distance_matrix(tree, max_distance=r, output_type='coo_matrix')
    if pairs.nnz > 0:  # If we have any pairs
        avg_distance = np.mean(pairs.data)
    else:
        avg_distance = r  # Default if no pairs found
    
    # 2. Clustering coefficient (average number of neighbors)
    # Count average number of neighbors within radius
    indices = list(zip(pairs.row, pairs.col))
    if len(indices) > 0:
        neighbors_count = np.bincount(pairs.row, minlength=len(positions))
        clustering = np.mean(neighbors_count)
    else:
        clustering = 0
    
    # 3. Group polarization (alignment of orientations)
    # This measures how well-aligned the particles are
    # Value close to 1 means particles are moving in the same direction
    cos_vals = np.cos(orientations)
    sin_vals = np.sin(orientations)
    mean_cos = np.mean(cos_vals)
    mean_sin = np.mean(sin_vals)
    polarization = np.sqrt(mean_cos**2 + mean_sin**2)
    
    return avg_distance, clustering, polarization

def animate(i):
    global orient, pos, food_positions, food_creation_time, current_time
    global food_lifetimes, avg_lifetime, min_lifetime, max_lifetime
    global food_active, food_respawn_counters, food_amounts, food_plots
    global avg_cohesion, avg_clustering, avg_polarization
    
    # Update current time
    current_time += 1
    
    # Update respawn counters for inactive food
    food_respawn_counters[~food_active] += 1
    
    # Respawn food if counter reaches delay
    respawn_indices = np.where((~food_active) & (food_respawn_counters >= food_respawn_delay))[0]
    for idx in respawn_indices:
        food_active[idx] = True
        food_respawn_counters[idx] = 0
        food_creation_time[idx] = current_time
        food_amounts[idx] = food_starting_amount  # Reset food amount
        food_plots[idx].set_alpha(0.3)  # Reset opacity
    
    # Neighbor interactions
    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type='coo_matrix')
    data = np.exp(1j * orient[dist.col])
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
    
    # Calculate cohesion metrics
    avg_distance, clustering, polarization = calculate_cohesion_metrics(pos, orient, tree, cohesion_radius)
    cohesion_history.append(avg_distance)
    clustering_history.append(clustering)
    polarization_history.append(polarization)
    
    # Update moving averages
    avg_cohesion = np.mean(cohesion_history)
    avg_clustering = np.mean(clustering_history)
    avg_polarization = np.mean(polarization_history)
    
    # Filter active food positions for interactions
    active_food_positions = food_positions[food_active]
    
    if len(active_food_positions) > 0:  # Only process if there is active food
        # Food attraction: find closest food for each bird
        tree_food = cKDTree(active_food_positions, boxsize=[L, L])
        d_food, idx_food_local = tree_food.query(pos, k=1)
        
        # Map local indices back to global food indices
        global_food_indices = np.where(food_active)[0]
        idx_food = global_food_indices[idx_food_local]
        
        # Compute periodic vector to food
        f_pos = food_positions[idx_food]
        delta = (f_pos - pos + L/2) % L - L/2
        norms = np.linalg.norm(delta, axis=1)
        unit_food = delta / norms[:, None]
        
        # Combine neighbor alignment and food attraction
        S_total = S + food_attraction_strength * (unit_food[:, 0] + 1j * unit_food[:, 1])
        
        # Handle eating: consume food gradually
        eating_mask = d_food < eat_radius
        if np.any(eating_mask):
            eating_indices = idx_food[eating_mask]
            for idx in np.unique(eating_indices):
                if food_active[idx]:
                    # Gradually decrease food amount
                    food_amounts[idx] -= food_consumption_rate
                    
                    # Update circle transparency based on remaining food
                    alpha = 0.3 * max(0, food_amounts[idx])
                    food_plots[idx].set_alpha(alpha)
                    
                    # If food depleted, mark as inactive and record lifetime
                    if food_amounts[idx] <= 0:
                        # Calculate lifetime of eaten food
                        lifetime = current_time - food_creation_time[idx]
                        food_lifetimes.append(lifetime)
                        
                        # Deactivate food
                        food_active[idx] = False
                        food_plots[idx].set_alpha(0)  # Make invisible
                        
                        # Update statistics
                        if len(food_lifetimes) > 0:
                            avg_lifetime = np.mean(food_lifetimes)
                            min_lifetime = np.min(food_lifetimes)
                            max_lifetime = np.max(food_lifetimes)
    else:
        # If no active food, just use neighbor alignment
        S_total = S
    
    # Update orientations with noise
    orient = np.angle(S_total) + eta * np.random.uniform(-np.pi, np.pi, size=N)
    
    # Update positions
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0 * speed
    pos[:, 1] += sin * v0 * speed
    pos %= L
    
    # Update particle quiver plot
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, orient)
    
    # Update food patches
    for i in range(num_food):
        if food_active[i]:
            food_plots[i].set_center(food_positions[i])
    
    # Update statistics text
    active_count = np.sum(food_active)
    inactive_count = num_food - active_count
    stats_str = (f"Food - Active: {active_count}/{num_food} \nEaten: {len(food_lifetimes)}"
                f"\nLifetimes - Avg: {avg_lifetime:.1f} steps \nMin: {min_lifetime:.1f} \nMax: {max_lifetime:.1f}")
    stats_text.set_text(stats_str)
    
    # Update cohesion metrics text
    cohesion_str = (f"Group Cohesion Metrics:"
                   f"\nAvg Distance: {avg_cohesion:.2f}"
                   f"\nClustering: {avg_clustering:.2f}"
                   f"\nPolarization: {avg_polarization:.2f}")
    cohesion_text.set_text(cohesion_str)
    
    return [qv, stats_text, cohesion_text] + food_plots

anim = FuncAnimation(fig, animate, frames=200, interval=10, blit=True)
plt.suptitle("Flocking with Food Resources")
plt.show()

# When the simulation is done, print the final statistics
def print_final_stats():
    if len(food_lifetimes) > 0:
        print("\nFood Lifetime Statistics:")
        print(f"Total food items eaten: {len(food_lifetimes)}")
        print(f"Average lifetime: {np.mean(food_lifetimes):.2f} steps")
        print(f"Minimum lifetime: {np.min(food_lifetimes):.2f} steps")
        print(f"Maximum lifetime: {np.max(food_lifetimes):.2f} steps")
        print(f"Standard deviation: {np.std(food_lifetimes):.2f} steps")
        
        # Histogram of food lifetimes
        plt.figure(figsize=(10, 6))
        plt.hist(food_lifetimes, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Food Lifetime (steps)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Food Lifetimes')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Print cohesion metrics
    print("\nGroup Cohesion Statistics:")
    print(f"Average distance between particles: {avg_cohesion:.2f}")
    print(f"Average clustering coefficient: {avg_clustering:.2f}")
    print(f"Average polarization: {avg_polarization:.2f}")
    
    # Plot cohesion metrics over time
    if len(cohesion_history) > 0:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(list(cohesion_history), label='Average Distance')
        plt.ylabel('Distance')
        plt.title('Group Cohesion Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(list(clustering_history), label='Clustering', color='green')
        plt.ylabel('Neighbors')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(list(polarization_history), label='Polarization', color='red')
        plt.ylabel('Alignment')
        plt.xlabel('Time Steps')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Function to modify food parameters during simulation
def set_food_params(new_positions=None, new_respawn_delay=None, new_food_radius=None, 
                    new_consumption_rate=None, new_attraction_strength=None):
    """
    Update food parameters during simulation.
    - new_positions: numpy array of shape (num_food, 2) with new positions
    - new_respawn_delay: integer representing steps to wait before respawning
    - new_food_radius: float for the visual radius of food circles
    - new_consumption_rate: rate at which food is consumed when eaten
    - new_attraction_strength: strength of food attraction for particles
    """
    global food_sources, food_positions, food_respawn_delay, food_radius
    global food_consumption_rate, food_attraction_strength, food_plots
    
    if new_positions is not None:
        if new_positions.shape[0] != len(food_sources):
            print(f"Error: Expected {len(food_sources)} food sources, but got {new_positions.shape[0]}")
            return
        food_sources = new_positions.copy()
        food_positions = new_positions.copy()
        # Update circle positions
        for i in range(len(food_plots)):
            food_plots[i].set_center(food_positions[i])
    
    if new_respawn_delay is not None:
        food_respawn_delay = new_respawn_delay
        print(f"Food respawn delay set to {food_respawn_delay} steps")
    
    if new_food_radius is not None:
        food_radius = new_food_radius
        print(f"Food radius set to {food_radius}")
        # Update circle radii
        for circle in food_plots:
            circle.set_radius(food_radius)
    
    if new_consumption_rate is not None:
        food_consumption_rate = new_consumption_rate
        print(f"Food consumption rate set to {food_consumption_rate}")
    
    if new_attraction_strength is not None:
        food_attraction_strength = new_attraction_strength
        print(f"Food attraction strength set to {food_attraction_strength}")

# Uncomment to call this when you want to see the final statistics
print_final_stats()

# Example of changing parameters:
# set_food_params(new_respawn_delay=100, new_food_radius=3.0, new_consumption_rate=0.02)
# 
# # Example of changing food positions:
# new_positions = np.array([
#     [8.0, 8.0],
#     [16.0, 8.0],
#     [24.0, 8.0],
#     [8.0, 16.0],
#     [16.0, 16.0], 
#     [24.0, 16.0],
#     [8.0, 24.0],
#     [16.0, 24.0],
#     [24.0, 24.0],
#     [20.0, 20.0]
# ])
# set_food_params(new_positions=new_positions)