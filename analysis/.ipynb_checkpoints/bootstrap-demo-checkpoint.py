import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt

# Step 1: Generate a set of particles with random 2D positions
# np.random.seed(42)  # for reproducibility
num_particles = 1000
max_distance = 10

# Generate random 2D positions for particles
particle_positions = np.random.uniform(low=-max_distance, high=max_distance, size=(num_particles, 2))

plt.figure(dpi=80, figsize=(2, 2))
plt.scatter(particle_positions[:,0], particle_positions[:,1], s=1)

# Step 2: Bin the particles by their distances from the origin
# Calculate distances from the origin for each particle
distances = np.linalg.norm(particle_positions, axis=1)

# Define bins
num_bins = 10
bins = np.linspace(0, max_distance, num_bins + 1)

# Bin the particles
binned_indices = np.digitize(distances, bins)

# Step 3: Assign errors to each bin using scipy's bootstrap method
# def bootstrap_errors(data, n_bootstrap):
# Function to be used for bootstrapping
def bootstrap_function(data):
    return np.histogram(data, bins=bins)[0]

data = (distances,)

# subsample_size = 1000  # Adjust this according to your memory constraints
# subsample_indices = np.random.choice(len(distances), size=subsample_size, replace=False)
# data = (distances[subsample_indices],)


n_bootstrap = 10000  # Number of bootstrap iterations
# Perform bootstrap
result = bootstrap(data, bootstrap_function, n_resamples=n_bootstrap)

print(result.standard_error)

# Calculate mean and standard error of each bin
mean_counts = np.mean(result.bootstrap_distribution, axis=1)
std_errors = result.standard_error # np.std(result.bootstrap_distribution, axis=1)

# mean_counts, std_errors = bootstrap_errors(distances, n_bootstrap)

# Plotting
bin_centers = 0.5 * (bins[1:] + bins[:-1])
bin_width = bins[1] - bins[0]

plt.figure(dpi=100)
plt.bar(bin_centers, mean_counts, width=bin_width, yerr=std_errors, capsize=5, label="Bootstrap Sample")
plt.bar(bin_centers, np.histogram(data, bins=bins)[0], label="True Histogram")
plt.xlabel('Distance from Origin')
plt.ylabel('Number of Particles')
plt.title('Binned Particle Counts with Error Bars')
# plt.grid(True)
plt.legend()
plt.show()