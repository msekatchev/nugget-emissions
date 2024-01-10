import numpy as np
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import sys

if len(sys.argv) != 2:
    print("Need an extra string to specify which voxel coords file to use")
    sys.exit(1)

# example: ioni_gas
name = sys.argv[1]


location_grid_coords = "../data/FIRE/grid-coords.npy"
location_non_empty_voxel_coords = "../data/FIRE/non-empty-coords-"+name+".npy"
# name = location_non_empty_voxel_coords[30:][0:location_non_empty_voxel_coords[30:].rfind(".")]
location_save_grid_ids = "../data/FIRE/grid-ids--"+name+".npy"

grid_c = np.load(location_grid_coords)
print(f">> loaded in voxel grid coordinates from {location_grid_coords}")

non_empty_points = np.load(location_non_empty_voxel_coords)
print(f">> loaded in non-empty voxel coordinates from {location_non_empty_voxel_coords}")

print(f">> {name}")

# Parameters
batch_size = 100

# Function to calculate distances and update grid_ids for a given interval
def process_interval(start, end):
    grid_ids[start:end] = np.argmin(distance.cdist(grid_c[start:end], non_empty_points), axis=1)

# Create an array to store grid_ids
grid_ids = np.empty(len(grid_c), dtype=int)

# Create batches
batch_counter = np.arange(batch_size, len(grid_c), batch_size)
batch_counter[-1] = len(grid_c)

# Process batches concurrently
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_interval, interval_min, interval_max): (interval_min, interval_max)
               for interval_min, interval_max in zip([0] + list(batch_counter), batch_counter)}

    for future in concurrent.futures.as_completed(futures):
        interval_min, interval_max = futures[future]
        print(f"Working on batch [{interval_min}-{interval_max}] out of {len(grid_c)}, so {interval_max/len(grid_c)*100}% done.", end="\r")
        # print(f"Processed interval [{interval_min}:{interval_max}]")


np.save(location_save_grid_ids, grid_ids)
