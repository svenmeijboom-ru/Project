import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# Re-defining the necessary parts of simple_vicsek here to make this standalone.
class CFG:
    seed = 123
    L = 32.0
    N = 180
    NUM_FRAMES = 800
    W_ALIGN = 1.0
    R0 = 3
    v0 = 0.5
    ETA = 0.05
    CLUSTERED = True
    FOOD = True
    NF = 2
    EAT_RADIUS = 0.5
    FOOD_STRENGTH = 0.0
    RESPAWN_DELAY = 50
    METRICS = True
    PLOT_METRICS = False
    DEBUG = False

def set_seed(seed=123):
    np.random.seed(seed)

def initialize_particles(cfg):
    if cfg.CLUSTERED:
        if cfg.NF == 1:
            food_center = np.array([cfg.L / 2, cfg.L / 2])
            offset_above = 8.0
            cluster_center = (food_center + np.array([0.0, offset_above])) % cfg.L
        else:
            cluster_center = np.array([cfg.L / 2, cfg.L / 2])
        cluster_radius = 1.0
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

def compute_average_cohesion(food_strength, cfg_template, seed):
    cfg = copy.deepcopy(cfg_template)
    cfg.FOOD_STRENGTH = food_strength
    cfg.seed = seed
    set_seed(cfg.seed)
    pos, orient = initialize_particles(cfg)
    cohesion_history = []

    for _ in range(cfg.NUM_FRAMES):
        tree = cKDTree(pos, boxsize=[cfg.L, cfg.L])
        dist = tree.sparse_distance_matrix(tree, max_distance=cfg.R0, output_type='coo_matrix')
        orient_neighbors = orient[dist.col]
        alignment_vec = np.exp(1j * orient_neighbors)
        alignment_matrix = sparse.coo_matrix((alignment_vec, (dist.row, dist.col)), shape=(cfg.N, cfg.N))
        alignment_sum = np.squeeze(np.asarray(alignment_matrix.tocsr().sum(axis=1)))
        S_total = cfg.W_ALIGN * alignment_sum

        if cfg.FOOD:
            if cfg.NF == 2:
                offset = cfg.L * 0.25
                center_y = cfg.L / 2
                food_positions = np.array([
                    [cfg.L / 2 - offset, center_y],
                    [cfg.L / 2 + offset, center_y]
                ])
            else:
                food_positions = create_evenly_spaced_food(int(np.sqrt(cfg.NF)), int(np.sqrt(cfg.NF)), cfg.L)[:cfg.NF]
            tree_food = cKDTree(food_positions, boxsize=[cfg.L, cfg.L])
            d_food, idx_food = tree_food.query(pos, k=1)
            f_pos = food_positions[idx_food]
            delta_food = (f_pos - pos + cfg.L / 2) % cfg.L - cfg.L / 2
            norms = np.linalg.norm(delta_food, axis=1)
            unit_food = delta_food / (norms[:, None] + 1e-8)
            food_vec = unit_food[:, 0] + 1j * unit_food[:, 1]
            S_total += cfg.FOOD_STRENGTH * food_vec

        orient = np.angle(S_total) + cfg.ETA * np.random.uniform(-np.pi, np.pi, size=cfg.N)
        cos, sin = np.cos(orient), np.sin(orient)
        pos[:, 0] += cos * cfg.v0
        pos[:, 1] += sin * cfg.v0
        pos %= cfg.L

        mean_heading = np.mean(np.exp(1j * orient))
        cohesion = np.abs(mean_heading)
        cohesion_history.append(cohesion)

    return np.mean(cohesion_history)

cfg_template = CFG()
cfg_template.NUM_FRAMES = 800
n_trials = 10
food_strength_values = np.linspace(0.0, 3.0, 5)

print(food_strength_values)

# Define test cases: (label, FOOD, NF)
test_cases = [
    ("No Food", False, 0),
    ("1 Food Source", True, 1),
    ("2 Food Sources", True, 2),
    ("4 Food Sources", True, 4),
]

# Store results
all_results = []

for label, food_on, nf in test_cases:
    cfg_template.FOOD = food_on
    cfg_template.NF = nf
    cohesion_means = []
    cohesion_stds = []

    for strength in food_strength_values:
        trial_cohesions = []
        for trial in range(n_trials):
            seed = 123 + trial
            avg_cohesion = compute_average_cohesion(strength, cfg_template, seed)
            trial_cohesions.append(avg_cohesion)
        cohesion_means.append(np.mean(trial_cohesions))
        cohesion_stds.append(np.std(trial_cohesions))
    
    all_results.append((label, food_strength_values.copy(), cohesion_means, cohesion_stds))

# Plotting the results
plt.figure(figsize=(10, 6))
for label, x, y, yerr in all_results:
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, label=label)

plt.xlabel('Food Strength')
plt.ylabel('Average Cohesion')
plt.title('Cohesion vs. Food Strength for Different Food Source Settings')
plt.grid(True)
plt.legend(title="Configuration")
plt.tight_layout()
plt.show()