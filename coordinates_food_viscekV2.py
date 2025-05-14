import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# System parameters
L = 32.0  # System size
rho = 3.0  # Particle density
N = int(rho*L**2)  # Number of particles
print("N =", N)

# Particle movement parameters
r0 = 1.0  # Interaction radius
deltat = 1.0  # Time step
factor = 0.5
v0 = r0/deltat*factor  # Base speed
iterations = 10000
eta = 0.15  # Noise parameter

# Food environment parameters
food_count = 5  # Number of food sources
food_radius = 2.0  # Radius of influence for food
food_lifetime_max = 100  # Maximum lifetime of a food source
food_replenish_rate = 0.02  # Probability of new food appearing per step
food_attraction_strength = 1.5  # Base strength of food attraction

# Initialize particles with random positions and orientations
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)
urge = np.random.uniform(0.2, 0.8, size=N)  # Individual "hunger" levels

# Initialize food sources
food_positions = np.random.uniform(0, L, size=(food_count, 2))
food_lifetimes = np.random.randint(1, food_lifetime_max, size=food_count)
food_amounts = np.ones(food_count)  # How much food is at each source

# Create figure and initialize visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, L)
ax.set_ylim(0, L)

# Plot initial particles and food
qv = ax.quiver(pos[:, 0], pos[:, 1], np.cos(orient), np.sin(orient), 
               urge, clim=[0, 1], cmap='viridis')
food_plots = []
for i in range(food_count):
    circle = Circle(food_positions[i], food_radius, alpha=0.3, color='green')
    food_plots.append(ax.add_patch(circle))

# Add a colorbar to represent urge levels
cbar = fig.colorbar(qv, ax=ax)
cbar.set_label('Hunger Urge')

# Helper functions defined BEFORE the animate function
def periodic_distance(p1, p2, L):
    """Calculate distance with periodic boundary conditions"""
    dx = np.abs(p1[0] - p2[0])
    dy = np.abs(p1[1] - p2[1])
    dx = min(dx, L - dx)
    dy = min(dy, L - dy)
    return np.sqrt(dx**2 + dy**2)

def food_direction(pos, food_pos, L):
    """Calculate direction towards food with periodic boundary conditions"""
    dx = food_pos[0] - pos[0]
    dy = food_pos[1] - pos[1]
    
    # Account for periodic boundaries
    if dx > L/2:
        dx -= L
    elif dx < -L/2:
        dx += L
    
    if dy > L/2:
        dy -= L
    elif dy < -L/2:
        dy += L
        
    if dx == 0 and dy == 0:
        return 0  # No direction if we're exactly at the food
    
    return np.arctan2(dy, dx)

def animate(i):
    global orient, pos, food_positions, food_lifetimes, food_amounts, urge, food_plots
    
    if i % 10 == 0:
        print(f"Step {i}")
    
    # Update food environment
    # Decrease food lifetimes
    food_lifetimes -= 1
    
    # Remove expired food sources and create new ones
    for j in range(food_count):
        if food_lifetimes[j] <= 0 or food_amounts[j] <= 0.1:
            # Create a new food source
            food_positions[j] = np.random.uniform(0, L, size=2)
            food_lifetimes[j] = np.random.randint(1, food_lifetime_max)
            food_amounts[j] = 1.0
            
            # Update visualization
            food_plots[j].set_center(food_positions[j])
            food_plots[j].set_alpha(0.3)
        else:
            # Adjust alpha based on remaining food
            food_plots[j].set_alpha(0.3 * food_amounts[j])
    
    # Occasionally add random new food sources
    if np.random.random() < food_replenish_rate:
        idx = np.random.randint(0, food_count)
        food_positions[idx] = np.random.uniform(0, L, size=2)
        food_lifetimes[idx] = np.random.randint(1, food_lifetime_max)
        food_amounts[idx] = 1.0
        food_plots[idx].set_center(food_positions[idx])
        food_plots[idx].set_alpha(0.3)
    
    # Build KD-tree for neighbor finding with periodic boundaries
    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type='coo_matrix')
    
    # Calculate alignment influence (original Vicsek model logic)
    data = np.exp(orient[dist.col]*1j)
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    alignment_influence = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
    alignment_direction = np.angle(alignment_influence)
    
    # Calculate food influence for each particle
    food_influence = np.zeros(N, dtype=complex)
    for j in range(food_count):
        if food_lifetimes[j] > 0 and food_amounts[j] > 0.1:
            for k in range(N):
                dist_to_food = periodic_distance(pos[k], food_positions[j], L)
                if dist_to_food < food_radius:
                    # Particles closer to food are more strongly attracted
                    strength = food_attraction_strength * (1 - dist_to_food/food_radius) * urge[k]
                    direction = food_direction(pos[k], food_positions[j], L)
                    food_influence[k] += strength * np.exp(direction * 1j)
                    
                    # Consume some food if very close
                    if dist_to_food < 0.5:
                        food_amounts[j] -= 0.01 * urge[k]
                        # Decrease urge when food is consumed
                        urge[k] = max(0.1, urge[k] - 0.05)
    
    # Calculate food direction
    food_magnitude = np.abs(food_influence)
    # Avoid division by zero warning for particles with no food influence
    food_directions = np.zeros(N)
    mask = food_magnitude > 0
    food_directions[mask] = np.angle(food_influence[mask])
    
    # Combine alignment and food influences based on urge
    # Higher urge gives more weight to food direction
    final_direction = np.zeros(N)
    for k in range(N):
        if food_magnitude[k] > 0:
            # Weight between alignment and food seeking based on urge
            alignment_weight = 1 - urge[k]
            food_weight = urge[k]
            
            # Convert directions to vectors for proper averaging
            align_vec = np.array([np.cos(alignment_direction[k]), np.sin(alignment_direction[k])])
            food_vec = np.array([np.cos(food_directions[k]), np.sin(food_directions[k])])
            
            # Combine vectors
            combined_vec = alignment_weight * align_vec + food_weight * food_vec
            final_direction[k] = np.arctan2(combined_vec[1], combined_vec[0])
        else:
            # If no food influence, follow alignment
            final_direction[k] = alignment_direction[k]
    
    # Add noise to final direction
    orient = final_direction + eta * np.random.uniform(-np.pi, np.pi, size=N)
    
    # Increase urge over time (getting hungrier)
    #urge = np.minimum(urge + 0.005, 1.0)
    
    # Update positions based on new orientations
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0
    pos[:, 1] += sin * v0
    
    # Apply periodic boundary conditions
    pos[pos > L] -= L
    pos[pos < 0] += L
    
    # Update visualization
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, urge)
    return [qv] + food_plots

# Create animation - reduced iterations for testing
anim = FuncAnimation(fig, animate, np.arange(1, 200), interval=1, blit=True)
plt.title("Vicsek Model with Dynamic Food Environment")
plt.show()