import numpy as np
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import sys
from itertools import product
from multiprocessing import Pool

if len(sys.argv) != 2:
    print("Need an extra string to specify which voxel coords file to use")
    sys.exit(1)

# example: ioni_gas
name = sys.argv[1]

location_grid_coords = "../data/FIRE/grid-coords.npy"
location_non_empty_voxel_coords = "../data/FIRE/non-empty-coords-"+name+".npy"
# name = location_non_empty_voxel_coords[30:][0:location_non_empty_voxel_coords[30:].rfind(".")]
location_save_grid_ids = "../data/FIRE/grid-ids-----"+name+".npy"

print(f">> loading in voxel grid coordinates...")
grid_c = np.load(location_grid_coords)
print(f">> loaded in voxel grid coordinates from {location_grid_coords}")

print(f">> loading in non-empty voxel coordinates...")
non_empty_points = np.load(location_non_empty_voxel_coords)
non_empty_points_length = len(non_empty_points)
print(f">> loaded in non-empty voxel coordinates from {location_non_empty_voxel_coords}")

print(f">> {name}")


current_min_distances = np.ones(len(grid_c), dtype=int) * 20**10
grid_ids = np.empty(len(grid_c), dtype=int)

def process_interval(indexes):
    start, end, points_offset_start, points_offset_end = indexes[0][0], indexes[0][1], indexes[1][0], indexes[1][1]
    print(f">> grid batch: [{start} - {end}] \t point batch: [{points_offset_start} - {points_offset_end}]", end="\r")
    # calculate all of the distances between this batch of grid points and filled voxel points
    distances = distance.cdist(grid_c[start:end], non_empty_points[points_offset_start:points_offset_end])
    # identify the nearest voxels to each grid point
    proposed_min_distances = np.min(distances, axis=1)
    proposed_min_ids = np.argmin(distances, axis=1) + points_offset_start

    improved_indexes = proposed_min_distances<current_min_distances[start:end]
    
    current_min_distances[start:end][improved_indexes] = proposed_min_distances[improved_indexes]
    grid_ids[start:end][improved_indexes] = proposed_min_ids[improved_indexes]


grid_batch_size   = 20000#10000 # 3000
points_batch_size = 20000#10000 # 4000

# Create grid batches
batch_counter = np.arange(grid_batch_size, len(grid_c), grid_batch_size)
batch_counter[-1] = len(grid_c)

# create points batches
points_counter = np.arange(points_batch_size, len(non_empty_points), points_batch_size)
points_counter[-1] = len(non_empty_points)

grid_batch_intervals     = np.array([[0]+list(batch_counter[0:-1]),   list(batch_counter)]).transpose()
points_counter_intervals = np.array([[0]+list(points_counter[0:-1]), list(points_counter)]).transpose()

print(">> calculating all batch combinations...")
# all_combinations = np.array(list(product(grid_batch_intervals, points_counter_intervals)))
def generate_combinations(args):
    grid_batch_intervals, points_counter_intervals = args
    return np.array(list(product(grid_batch_intervals, points_counter_intervals)))

# Define the number of processes based on your system's capabilities
num_processes = 16

# Split the workload
chunk_size = len(grid_batch_intervals) // num_processes
chunks = [(grid_batch_intervals[i:i+chunk_size], points_counter_intervals) for i in range(0, len(grid_batch_intervals), chunk_size)]

# Use multiprocessing Pool to parallelize the combination generation
with Pool(processes=num_processes) as pool:
    all_combinations = np.concatenate(pool.map(generate_combinations, chunks))

# with multithreading
print(f">> grid batch max: {len(grid_c)} \t points batch max: {non_empty_points_length}")

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_interval, indexes): indexes
               for indexes in all_combinations}

np.save(location_save_grid_ids, grid_ids)
