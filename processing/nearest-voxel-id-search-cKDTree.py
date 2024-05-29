import numpy as np
import sys
from scipy.spatial import cKDTree
import timeit

if len(sys.argv) != 2:
    print("Need an extra string to specify which voxel coords file to use")
    sys.exit(1)

# example: ioni_gas
name = sys.argv[1]

location_grid_coords = "../data/FIRE/grid-coords-uv.npy"
location_non_empty_voxel_coords = "../data/FIRE/non-empty-coords-"+name+"-uv.npy"
# name = location_non_empty_voxel_coords[30:][0:location_non_empty_voxel_coords[30:].rfind(".")]
location_save_grid_ids = "../data/FIRE/grid-ids--"+name+"-cKDTree-uv.npy"

print(f">> loading in voxel grid coordinates...")
grid_c = np.load(location_grid_coords)
print(f">> loaded in voxel grid coordinates from {location_grid_coords}")

print(f">> loading in non-empty voxel coordinates...")
non_empty_points = np.load(location_non_empty_voxel_coords)
print(f">> loaded in non-empty voxel coordinates from {location_non_empty_voxel_coords}")

print(f">> {name}")


grid_ids = np.empty(len(grid_c), dtype=int)

t0 = timeit.default_timer()
print(f">> generating cKDTree")
tree_points = cKDTree(non_empty_points)
print(f">> started cKDTree query")
grid_ids = tree_points.query(grid_c)[1]
tf = timeit.default_timer()

np.save(location_save_grid_ids, grid_ids)
