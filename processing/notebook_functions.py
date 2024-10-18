import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from astropy import constants as cst
from astropy import units as u
import healpy as hp

from aqn import *
from constants import *
from survey_parameters import *
from skymap_plotting_functions import *
from aqn_simulation import *

import sys
sys.path.append('../analysis')

# from mcmc_models import *

import logging

# Set up the logger
logger = logging.getLogger('debug_logger')
logger.setLevel(logging.DEBUG)

# Check if handlers already exist
if not logger.hasHandlers():
    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

# Example debug log messages
# logger.debug('This is a debug message')

def save_class(c, location, name):
    with open(location + name + ".pkl", 'wb') as file:
        pickle.dump(c, file)

def load_class(location, name):
    with open(location + name + ".pkl", 'rb') as file:
        return pickle.load(file)



class cube_generation:
    def __init__(self):
        logger.debug(">> importing FIRE simulation data")
        with open('../data/FIRE/part.pkl', 'rb') as file: self.part = pickle.load(file)
        logger.debug(">> imported  FIRE simulation data")

        # define cube resolution parameters
        self.cube_length       = 20   # kpc # 20 for radio
        self.voxel_resolution  = 2**9 # bins / kpc
        # logger.debug(">> voxels per length: {self.voxel_resolution}")
        
        self.voxel_length = self.cube_length / self.voxel_resolution * u.kpc
        self.voxel_volume = self.voxel_length**3
        
        # create empty dictionaries for dark matter and gas components
        self.dark_mat, self.neut_gas, self.ioni_gas = {}, {}, {}
        self.dark_mat["name"], self.neut_gas["name"], self.ioni_gas["name"] = "Dark Matter", "Neutral Gas", "Ionized Gas"
        self.dark_mat["short-name"], self.neut_gas["short-name"], self.ioni_gas["short-name"] = "dark_mat", "neut_gas", "ioni_gas"
    
    def fix_rotation(self):
        logger.debug('>> correcting Galaxy orientation')
        
        self.gas = {}
        self.gas["masses"] = self.part["gas"].prop("mass")
        self.gas["coords"] = self.part['gas'].prop('host.distance')
        
        self.gas["x"], self.gas["y"], self.gas["z"] = self.gas["coords"][:,0], self.gas["coords"][:,1], self.gas["coords"][:,2]
        self.gas["r"] = np.sqrt(self.gas["x"]**2 + self.gas["y"]**2 + self.gas["z"]**2)
        
        # gas["region"] = np.where(gas["r"] < 5)  
        self.gas["region"] = np.where((self.gas["r"] < 9.2) & (self.gas["r"] > 7.6))  
        
        for i in ["masses", "x", "y", "z", "r", "coords"]:
            self.gas[i] = self.gas[i][self.gas["region"]]
        
        from scipy.spatial.transform import Rotation as R
        
        # Assume gas is your dictionary with coordinates and masses
        x = self.gas["x"]
        y = self.gas["y"]
        z = self.gas["z"]
        masses = self.gas["masses"]
        
        # Compute the square root of the masses for weighting
        weights = np.sqrt(masses)
        
        # Construct the weighted matrix A
        A = np.vstack([x * weights, y * weights, weights]).T
        
        # Apply weights to z as well
        z_weighted = z * weights
        
        # Solve for the coefficients a, b, c with weighted least squares
        coefficients, residuals, _, _ = np.linalg.lstsq(A, z_weighted, rcond=None)
        
        a, b, c = coefficients
        
        # The normal vector to the plane
        normal_vector = np.array([a, b, -1])
        normal_vector /= np.linalg.norm(normal_vector)
        
        # Calculate the axis of rotation (cross product with z-axis)
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(normal_vector, z_axis)
        rotation_axis /= np.linalg.norm(rotation_axis)
        
        # Calculate the angle of rotation
        theta = np.arccos(np.dot(normal_vector, z_axis))
        
        # Construct the rotation matrix
        self.rotation = R.from_rotvec(rotation_axis * theta)
        rotation_matrix = self.rotation.as_matrix()
        
        # Apply the rotation to the galaxy coordinates
        rotated_coordinates = self.rotation.apply(np.vstack([x, y, z]).T)
        
        # Update the gas dictionary with the new rotated coordinates
        self.gas["x"], self.gas["y"], self.gas["z"] = rotated_coordinates.T
        self.gas["coords"] = self.rotation.apply(self.gas["coords"])


    def prepare_particles(self):
        logger.debug('>> extracting particle mass data')
        
        self.gas_temp = self.part['gas'].prop('temperature')
        
        self.efrac = self.part["gas"].prop("electron.fraction")
        self.efrac = self.efrac / np.max(self.efrac)
        
        self.dark_mat["masses"] = self.part["dark"].prop("mass")
        self.neut_gas["masses"] = self.part["gas"].prop("mass") * (1-self.efrac)
        self.ioni_gas["masses"] = self.part["gas"].prop("mass") * self.efrac
        
        # obtain velocities of all particles
        self.dark_mat["v"] = self.part['dark'].prop('velocity')
        self.neut_gas["v"] = self.part['gas'].prop('velocity')
        self.ioni_gas["v"] = self.part['gas'].prop('velocity')
        
        # obtain coordinates of all particles within cube relative to Milky Way center
        self.dark_mat["coords"] = self.part['dark'].prop('host.distance')
        self.neut_gas["coords"] = self.part['gas'].prop('host.distance')
        self.ioni_gas["coords"] = self.part['gas'].prop('host.distance')
        
        for dictt in [self.dark_mat, self.ioni_gas, self.neut_gas]:
            dictt["coords"] = self.rotation.apply(dictt["coords"])
            dictt["v"]      = self.rotation.apply(dictt["v"])

        logger.debug('>> creating cube coordinates filters')
        
        self.dark_mat["within_cube"] = np.where((np.abs(self.dark_mat["coords"][:,0]) < self.cube_length/2) & 
                                                (np.abs(self.dark_mat["coords"][:,1]) < self.cube_length/2) &
                                                (np.abs(self.dark_mat["coords"][:,2]) < self.cube_length/2))
        self.neut_gas["within_cube"] = np.where((np.abs(self.neut_gas["coords"][:,0]) < self.cube_length/2) & 
                                                (np.abs(self.neut_gas["coords"][:,1]) < self.cube_length/2) &
                                                (np.abs(self.neut_gas["coords"][:,2]) < self.cube_length/2))
        self.ioni_gas["within_cube"] = np.where((np.abs(self.ioni_gas["coords"][:,0]) < self.cube_length/2) & 
                                                (np.abs(self.ioni_gas["coords"][:,1]) < self.cube_length/2) &
                                                (np.abs(self.ioni_gas["coords"][:,2]) < self.cube_length/2))
        
        for dictt in [self.dark_mat, self.ioni_gas, self.neut_gas]:
            dictt["masses"] = dictt["masses"][dictt["within_cube"]]
            dictt["v"] = dictt["v"][dictt["within_cube"]]
            # dictt["coords"] = dictt["masses"][dictt["within_cube"]]


    def bin_mass_and_velocity(self):
        # create bins based on defined resolution parameters
        self.bins = np.linspace(-self.cube_length/2,self.cube_length/2,self.voxel_resolution+1)
        # bin center coordinates will be used to identify voxels
        self.bin_centers = (self.bins[1:] + self.bins[:-1])/2
        self.voxel_centers = np.array([self.bin_centers, self.bin_centers, self.bin_centers]) # kpc

        # use histograms to obtain mass counts and velocities within each voxel
        logger.debug('>> binning masses and velocities')
        for dictt in [self.dark_mat, self.neut_gas, self.ioni_gas]:
        
            
            # bin all particle masses within cube, weighing by their mass
            dictt["mass_count"], bin_edges, bin_numbers = stats.binned_statistic_dd(dictt["coords"][dictt["within_cube"]],
                                  dictt["masses"],
                                  statistic='sum',
                                  bins=(self.bins,self.bins,self.bins),
                                  expand_binnumbers=True)
            logger.debug('>> >> '+ dictt["short-name"] + " mass done")
            # velocity calculations are currently broken -- should be done after calculating the voronoied density cubes
            # bin all particle velocities within cube, weighing by their mass
            velocities, bin_edges, bin_numbers = stats.binned_statistic_dd(dictt["coords"][dictt["within_cube"]],
                                  [dictt["v"][:,0], dictt["v"][:,1], dictt["v"][:,2]]*dictt["masses"],
                                  statistic='sum',
                                  bins=(self.bins,self.bins,self.bins),
                                  expand_binnumbers=True,
                                  binned_statistic_result=None)
            
            dictt["v_x"], dictt["v_y"], dictt["v_z"] = velocities[0] * u.km/u.s, velocities[1] * u.km/u.s, velocities[2] * u.km/u.s
            logger.debug('>> >> '+ dictt["short-name"] + " velocity done")
            
            # finish weighed average calculation
            non_empty_v_x, non_empty_v_y, non_empty_v_z = velocities[0]!=0, velocities[1]!=0, velocities[2]!=0    
            dictt["v_x"][non_empty_v_x] = velocities[0][non_empty_v_x] / dictt["mass_count"][non_empty_v_x] * u.km/u.s
            dictt["v_y"][non_empty_v_y] = velocities[1][non_empty_v_y] / dictt["mass_count"][non_empty_v_y] * u.km/u.s
            dictt["v_z"][non_empty_v_z] = velocities[2][non_empty_v_z] / dictt["mass_count"][non_empty_v_z] * u.km/u.s

    
    def bin_temperature(self):
        logger.debug('>> binning ioni_gas temperatures, weighed by mass')
    
        # bin all ionized gas particle temperatures within cube, weighing by their mass
        temperatures, bin_edges, bin_numbers = stats.binned_statistic_dd(self.ioni_gas["coords"][self.ioni_gas["within_cube"]], 
                              self.gas_temp[self.ioni_gas["within_cube"]]*self.ioni_gas["masses"], 
                              statistic='sum', 
                              bins=(self.bins,self.bins,self.bins),
                              expand_binnumbers=True, 
                              binned_statistic_result=None)
        
        self.ioni_gas["temperatures"] = temperatures * u.K
        non_empty_temp = temperatures != 0
        self.ioni_gas["temperatures"][non_empty_temp] = temperatures[non_empty_temp] / self.ioni_gas["mass_count"][non_empty_temp] * u.K
    
    def calculate_densities(self):
        # calculate densities from mass counts
        logger.debug(">> calculating density voxels from mass count voxels")
        print(f"---", end="\r")
        self.dark_mat["density"] = (self.dark_mat["mass_count"] * u.solMass).to(u.kg) / self.voxel_volume.to(u.m**3)
        print(f"#--", end="\r")
        self.neut_gas["density"] = (self.neut_gas["mass_count"] * u.solMass).to(u.kg) / self.voxel_volume.to(u.m**3)
        print(f"##-", end="\r")
        self.ioni_gas["density"] = (self.ioni_gas["mass_count"] * u.solMass).to(u.kg) / self.voxel_volume.to(u.m**3)
        print(f"###")
    
    def calculate_dvs(self):
        # calculate the delta in velocities between dark matter and visible matter
        logger.debug(">> calculating velocity differences between dark matter and visible matter")
        print(f"--", end="\r")
        self.dv_neut = np.sqrt((self.dark_mat["v_x"] - self.neut_gas["v_x"])**2 + \
                               (self.dark_mat["v_y"] - self.neut_gas["v_y"])**2 + \
                               (self.dark_mat["v_z"] - self.neut_gas["v_z"])**2).to(u.m/u.s)
        print(f"#-", end="\r")
        self.dv_ioni = np.sqrt((self.dark_mat["v_x"] - self.ioni_gas["v_x"])**2 + \
                               (self.dark_mat["v_y"] - self.ioni_gas["v_y"])**2 + \
                               (self.dark_mat["v_z"] - self.ioni_gas["v_z"])**2).to(u.m/u.s)
        print(f"##")

    def prepare_for_voronoi(self):
        logger.debug(">> saving files for voronoi voxelization's nearest neighbour search")
        
        xx_c = np.meshgrid(self.bin_centers,self.bin_centers, self.bin_centers)[0]#.astype(np.float32)
        yy_c = np.meshgrid(self.bin_centers,self.bin_centers, self.bin_centers)[1]#.astype(np.float32)
        zz_c = np.meshgrid(self.bin_centers,self.bin_centers, self.bin_centers)[2]#.astype(np.float32)
        
        # These two are equivalent:
        # grid_c = np.dstack(np.array([xx_c,yy_c, zz_c])).reshape(-1,3)
        self.grid_c = np.vstack((xx_c.ravel(), yy_c.ravel(), zz_c.ravel())).T
        np.save('../data/FIRE/grid-coords.npy', self.grid_c)
        logger.debug(">>\tgrid coordinates")
        
        logger.debug(">>\tdensities")
        for dictt in [self.dark_mat, self.neut_gas, self.ioni_gas]:
            non_empty_ioni_gas = dictt["density"]!=0
            non_empty_points = np.array([xx_c[non_empty_ioni_gas], yy_c[non_empty_ioni_gas], zz_c[non_empty_ioni_gas]]).transpose()
        
            save_file =  '../data/FIRE/non-empty-coords-'+dictt["short-name"]+'.npy'
            np.save(save_file, non_empty_points)
        
        logger.debug(">>\tionized gas temperature")
        non_empty_ioni_gas = self.ioni_gas["temperatures"]!=0
        non_empty_points = np.array([xx_c[non_empty_ioni_gas], yy_c[non_empty_ioni_gas], zz_c[non_empty_ioni_gas]]).transpose()
        save_file =  '../data/FIRE/non-empty-coords-'+'ioni_gas-temp'+'.npy'
        np.save(save_file, non_empty_points)
        
        logger.debug(">>\trelative velocity, ionized gas vs dark matter")
        non_empty_ioni_gas = self.dv_ioni!=0
        non_empty_points = np.array([xx_c[non_empty_ioni_gas], yy_c[non_empty_ioni_gas], zz_c[non_empty_ioni_gas]]).transpose()
        save_file =  '../data/FIRE/non-empty-coords-'+'dv_ioni'+'.npy'
        np.save(save_file, non_empty_points)
        
        logger.debug(">>\trelative velocity, neutral gas vs dark matter")
        non_empty_ioni_gas = self.dv_neut!=0
        non_empty_points = np.array([xx_c[non_empty_ioni_gas], yy_c[non_empty_ioni_gas], zz_c[non_empty_ioni_gas]]).transpose()
        save_file =  '../data/FIRE/non-empty-coords-'+'dv_neut'+'.npy'
        np.save(save_file, non_empty_points)

        self.voro = voronoi(self)

    def compute_voronoi_cubes(self):

        voro = self.voro
        
        counter = np.array([0])
        m = 30
        def advance():
            print("#"*counter[0] + "-"*(m-counter[0]), end="\r")
            counter[0] = counter[0] + 1  
            
        self.cubes = {}
        
        # Final results:
        advance()
        grid_ids_dark = np.load('../data/FIRE/grid-ids--dark_mat-cKDTree.npy').astype(int)
        advance()
        grid_ids_ioni = np.load('../data/FIRE/grid-ids--ioni_gas-cKDTree.npy').astype(int)
        advance()
        grid_ids_neut = np.load('../data/FIRE/grid-ids--neut_gas-cKDTree.npy').astype(int)
        advance()
        
        # ionized gas density
        self.cubes["ioni_gas_density"] = voro.compute_voronoi_cube_density(self.grid_c, grid_ids_ioni, self.ioni_gas)
        advance()
        voro.plot_pre_voronoi(self.ioni_gas["density"],         'Density  [kg$\cdot$m$^{-3}$]', "../visuals/voronoi-final-ioni_gas-density-pre")
        advance()
        voro.plot_post_voronoi(self.cubes["ioni_gas_density"],  'Density  [kg$\cdot$m$^{-3}$]', "../visuals/voronoi-final-ioni_gas-density-post")
        advance()
        
        # ionized gas temperature
        self.cubes["ioni_gas_temp"] = voro.compute_voronoi_cube_temp(self.grid_c, grid_ids_ioni, self.ioni_gas)
        advance()
        voro.plot_pre_voronoi(self.ioni_gas["temperatures"],   'Temperature [K]', "../visuals/voronoi-final-ioni_gas-temp-pre")
        advance()
        voro.plot_post_voronoi(self.cubes["ioni_gas_temp"],    'Temperature [K]', "../visuals/voronoi-final-ioni_gas-temp-post")
        advance()
        
        # dark matter density
        self.cubes["dark_mat_density"] = voro.compute_voronoi_cube_density(self.grid_c, grid_ids_dark, self.dark_mat)
        advance()
        voro.plot_pre_voronoi(self.dark_mat["density"],         'Density  [kg$\cdot$m$^{-3}$]', "../visuals/voronoi-final-dark_mat-density-pre")
        advance()
        voro.plot_post_voronoi(self.cubes["dark_mat_density"],  'Density  [kg$\cdot$m$^{-3}$]', "../visuals/voronoi-final-dark_mat-density-post")
        advance()
        
        # neutral gas density
        self.cubes["neut_gas_density"] = voro.compute_voronoi_cube_density(self.grid_c, grid_ids_neut, self.neut_gas)
        advance()
        voro.plot_pre_voronoi(self.neut_gas["density"],         'Density  [kg$\cdot$m$^{-3}$]', "../visuals/voronoi-final-neut_gas-density-pre")
        advance()
        voro.plot_post_voronoi(self.cubes["neut_gas_density"],  'Density  [kg$\cdot$m$^{-3}$]', "../visuals/voronoi-final-neut_gas-density-post")
        advance()
        
        # change in velocities
        dark_mat_x = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_dark, self.dark_mat, "v_x")
        advance()
        dark_mat_y = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_dark, self.dark_mat, "v_y")
        advance()
        dark_mat_z = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_dark, self.dark_mat, "v_z")
        advance()
        
        ioni_gas_x = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_ioni, self.ioni_gas, "v_x")
        advance()
        ioni_gas_y = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_ioni, self.ioni_gas, "v_y")
        advance()
        ioni_gas_z = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_ioni, self.ioni_gas, "v_z")
        advance()
        
        neut_gas_x = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_neut, self.neut_gas, "v_x")
        advance()
        neut_gas_y = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_neut, self.neut_gas, "v_y")
        advance()
        neut_gas_z = voro.compute_voronoi_cube_vel(self.grid_c, grid_ids_neut, self.neut_gas, "v_z")
        advance()
        
        self.cubes["dv_neut"] = np.sqrt((dark_mat_x - neut_gas_x)**2 + (dark_mat_y - neut_gas_y)**2 + (dark_mat_z - neut_gas_z)**2).to(u.m/u.s)
        advance()
        self.cubes["dv_ioni"] = np.sqrt((dark_mat_x - ioni_gas_x)**2 + (dark_mat_y - ioni_gas_y)**2 + (dark_mat_z - ioni_gas_z)**2).to(u.m/u.s)
        advance()
        voro.plot_pre_voronoi(self.dv_neut,             'Velocity  [m$\cdot$s$^{-1}$]', "../visuals/voronoi-final-dv_neut-pre")
        advance()
        voro.plot_post_voronoi(self.cubes["dv_neut"],   'Velocity  [m$\cdot$s$^{-1}$]', "../visuals/voronoi-final-dv_neut-post")
        advance()
        voro.plot_pre_voronoi(self.dv_ioni,             'Velocity  [m$\cdot$s$^{-1}$]', "../visuals/voronoi-final-dv_ioni-pre")
        advance()
        voro.plot_post_voronoi(self.cubes["dv_ioni"],   'Velocity  [m$\cdot$s$^{-1}$]', "../visuals/voronoi-final-dv_ioni-post")
        advance()
        
        # change in velocities

class voronoi:
    def __init__(self, fire):
        self.voxel_resolution = fire.voxel_resolution
        self.voxel_centers = fire.voxel_centers

    def plot_pre_voronoi(self, x, unit_label, save_name):
        z_slice_min, z_slice_max = [-0.5,0.5]
        mask = (self.voxel_centers[2] >= z_slice_min) & (self.voxel_centers[2] < z_slice_max)
        slice_count = np.sum(x[:,:,mask], axis=2).value
        slice_count[slice_count==0] = np.min(slice_count[slice_count>0])
        plt.figure(dpi=500)
        plt.imshow(slice_count, extent=[-10, 10, -10, 10], norm = matplotlib.colors.LogNorm())
        
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(unit_label, fontsize=20)
        plt.xlabel('$x$ [kpc]', size=20)
        plt.ylabel('$y$ [kpc]', size=20)
        plt.xticks(np.array([-10,-5,0,5,10]),fontsize=15)
        plt.yticks(np.array([-10,-5,0,5,10]), fontsize=15)
        
        plt.title("Pre-voronoi", size=20)
        
        # plt.savefig(save_name+".png", bbox_inches='tight')
        # plt.savefig(save_name+".svg", bbox_inches='tight')
        
        plt.show()
        # plt.close()
    
    def plot_post_voronoi(self, x, unit_label, save_name):
        z_slice_min, z_slice_max = [-0.5, 0.5]
        mask = (self.voxel_centers[2] >= z_slice_min) & (self.voxel_centers[2] < z_slice_max)
        slice_count = np.average(x[:,:,mask], axis=2).value
        # slice_count[slice_count==0] = np.min(slice_count[slice_count>0])
        plt.figure(dpi=500)
        plt.imshow(slice_count, extent=[-10, 10, -10, 10], norm = matplotlib.colors.LogNorm())
        
        
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(unit_label, fontsize=20)
        plt.xlabel('$x$ [kpc]', size=20)
        plt.ylabel('$y$ [kpc]', size=20)
        plt.xticks(np.array([-10,-5,0,5,10]),fontsize=15)
        plt.yticks(np.array([-10,-5,0,5,10]), fontsize=15)
        
        plt.title("Post-voronoi", size=20)
        
        # plt.savefig(save_name+".png", bbox_inches='tight')
        # plt.savefig(save_name+".svg", bbox_inches='tight')
        
        # plt.close()
        plt.show()
    
    def compute_voronoi_cube_density(self, grid_c, grid_ids, dictt):
        non_empty_points = dictt["density"] != 0
        
        non_empty_points_reshaped = np.reshape(non_empty_points, self.voxel_resolution**3)
        quantity = np.reshape(dictt["density"], self.voxel_resolution**3)
    
        # divide by the bincount to average the density across the voronoi volumes
        voronoied_result = quantity[non_empty_points_reshaped][grid_ids] / np.bincount(grid_ids)[grid_ids]
        voronoied_result = voronoied_result.reshape((self.voxel_resolution, self.voxel_resolution, self.voxel_resolution))
    
        return voronoied_result
    
    def compute_voronoi_cube_temp(self, gric_c, grid_ids, dictt):
        non_empty_points = dictt["temperatures"] != 0
        
        non_empty_points_reshaped = np.reshape(non_empty_points, self.voxel_resolution**3)
        quantity = np.reshape(dictt["temperatures"], self.voxel_resolution**3)
    
        # don't divide by the bincount here, we don't want to average across voronoi volumes for the temperatures
        voronoied_result = quantity[non_empty_points_reshaped][grid_ids] # / np.bincount(grid_ids)[grid_ids]
        voronoied_result = voronoied_result.reshape((self.voxel_resolution, self.voxel_resolution, self.voxel_resolution))
    
        return voronoied_result    
    
    def compute_voronoi_cube_vel(self, gric_c, grid_ids, dictt, dir = "v_x"):
        non_empty_points = dictt[dir] != 0
        
        non_empty_points_reshaped = np.reshape(non_empty_points, self.voxel_resolution**3)
        quantity = np.reshape(dictt[dir], self.voxel_resolution**3)
    
        # don't divide by the bincount here, we don't want to average across voronoi volumes for the temperatures
        voronoied_result = quantity[non_empty_points_reshaped][grid_ids] # / np.bincount(grid_ids)[grid_ids]
        voronoied_result = voronoied_result.reshape((self.voxel_resolution, self.voxel_resolution, self.voxel_resolution))
    
        return voronoied_result




def import_cubes(reference=False, location="../data/FIRE/"):
    # grid_coords = np.load("../data/FIRE/grid-coords.npy") # !!! these are now rotated!

    cubes = {}
    
    cubes["aqn_emit"] = np.load(location + "cubes/cube-aqn_emit.npy") * u.photon / u.cm**3 / u.s / u.Angstrom / u.sr
    cubes["dark_mat"] = np.load(location + "cubes/cube-dark_mat_density.npy") * u.kg/u.m**3 * 2/5
    cubes["ioni_gas"] = np.load(location + "cubes/cube-ioni_gas_density.npy") * u.kg/u.m**3
    cubes["neut_gas"] = np.load(location + "cubes/cube-neut_gas_density.npy") * u.kg/u.m**3
    cubes["temp_ion"] = np.load(location + "cubes/cube-ioni_gas_temp.npy") * u.K
    cubes["dv_ioni"]  = np.load(location + "cubes/cube-dv_ioni.npy") * u.m/u.s
    cubes["dv_neut"]  = np.load(location + "cubes/cube-dv_neut.npy") * u.m/u.s
    
    # perform some unit conversions
    cubes["ioni_gas"] = (cubes["ioni_gas"]/cst.m_p.si).to(1/u.cm**3)
    cubes["neut_gas"] = (cubes["neut_gas"]/cst.m_p.si).to(1/u.cm**3)
    cubes["dark_mat"] = (cubes["dark_mat"]) # /  m_aqn_kg).to(1/u.cm**3) # done in code
    cubes["temp_ion"] =  cubes["temp_ion"]*K_to_eV
    cubes["dv_ioni"]  =  cubes["dv_ioni"] /cst.c
    cubes["dv_neut"]  =  cubes["dv_neut"] /cst.c
    cubes["temp_ion"] = cubes["temp_ion"] + \
                        1/2 * (938*u.MeV).to(u.eV) * cubes["dv_ioni"]**2
    
    if reference:
        cubes["dark_mat_ref"] = np.load(location + "cubes/cube-dark_mat_density.npy") * u.kg/u.m**3 * 2/5
        cubes["ioni_gas_ref"] = np.load(location + "cubes/cube-ioni_gas_density.npy") * u.kg/u.m**3
        cubes["neut_gas_ref"] = np.load(location + "cubes/cube-neut_gas_density.npy") * u.kg/u.m**3

        # cubes["ioni_gas_ref"] = (cubes["ioni_gas_ref"]/cst.m_p.si).to(1/u.cm**3)
        # cubes["neut_gas_ref"] = (cubes["neut_gas_ref"]/cst.m_p.si).to(1/u.cm**3)
        # cubes["dark_mat_ref"] = (cubes["dark_mat_ref"]) # /  m_aqn_kg).to(1/u.cm**3) # done in code

        
    return cubes



def epsilon_parameter_relations_study(quant, m_aqn_kg, frequency_band):
    parameter_relations_save_location = "../visuals/parameter_relations/"

    velocity_array = np.linspace(20, 800, 100) * u.km/u.s
    epsilon_array = parameter_variation(quant, m_aqn_kg, frequency_band, "dv_ioni", velocity_array)

    fig, ax = plot_parameter_variation(r"$\Delta v$ [km/s]", r"$\epsilon$ ["+str(epsilon_units.unit)+"]", velocity_array, epsilon_array)
    plot_scaling_relation(ax, velocity_array, 13/7, np.max(epsilon_array))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(parameter_relations_save_location+"epsilon_vs_dv.png", bbox_inches="tight")
    plt.show()

    ioni_gas_array = np.linspace(1e-4, 1e-1, 10) * 1/u.cm**3
    epsilon_array = parameter_variation(quant, m_aqn_kg, frequency_band, "ioni_gas", ioni_gas_array)

    fig, ax = plot_parameter_variation(r"$n_{ion}$ [1/cm$^3$]", r"$\epsilon$ ["+str(epsilon_units.unit)+"]", ioni_gas_array, epsilon_array)
    plot_scaling_relation(ax, ioni_gas_array, 13/7, np.min(epsilon_array))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(parameter_relations_save_location+"epsilon_vs_ioni_gas.png", bbox_inches="tight")
    plt.show()

    m_aqn_kg_array = np.linspace(1e-4, 1e-1, 10) * u.kg
    epsilon_array = parameter_variation(quant, m_aqn_kg, frequency_band, "aqn_mass", m_aqn_kg_array, True)

    fig, ax = plot_parameter_variation(r"$m_{aqn}$ [kg]", r"$\epsilon$ ["+str(epsilon_units.unit)+"]", m_aqn_kg_array, epsilon_array)
    plot_scaling_relation(ax, m_aqn_kg_array, 19/21, np.min(epsilon_array))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(parameter_relations_save_location+"epsilon_vs_m_aqn.png", bbox_inches="tight")
    plt.show()

    T_gas_array = np.linspace(1e2, 1e6, 100) * u.K
    epsilon_array = parameter_variation(quant, m_aqn_kg, frequency_band, "temp_ion", T_gas_array)

    fig, ax = plot_parameter_variation(r"$T_{gas}$ [K]", r"$\epsilon$ ["+str(epsilon_units.unit)+"]", T_gas_array, epsilon_array)
    plot_scaling_relation(ax, T_gas_array, 26/7, np.min(epsilon_array))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(parameter_relations_save_location+"epsilon_vs_t_gas.png", bbox_inches="tight")
    plt.show()

def t_aqn_parameter_relations_study(quant, m_aqn_kg, frequency_band):
    parameter_relations_save_location = "../visuals/parameter_relations/"

    velocity_array = np.linspace(20, 800, 100) * u.km/u.s
    epsilon_array = parameter_variation_t_aqn(quant, m_aqn_kg, frequency_band, "dv_ioni", velocity_array)
    epsilon_array = epsilon_array / K_to_eV
    fig, ax = plot_parameter_variation(r"$\Delta v$ [km/s]", r"$T_{aqn}$", velocity_array, epsilon_array)
    plot_scaling_relation(ax, velocity_array, 4/7, np.max(epsilon_array))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(parameter_relations_save_location+"t_aqn_vs_dv.png", bbox_inches="tight")
    plt.show()

    ioni_gas_array = np.linspace(1e-4, 1e-1, 10) * 1/u.cm**3
    epsilon_array = parameter_variation_t_aqn(quant, m_aqn_kg, frequency_band, "ioni_gas", ioni_gas_array)

    fig, ax = plot_parameter_variation(r"$n_{ion}$ [1/cm$^3$]", r"$T_{aqn}$", ioni_gas_array, epsilon_array)
    plot_scaling_relation(ax, ioni_gas_array, 4/7, np.min(epsilon_array))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(parameter_relations_save_location+"t_aqn_vs_ioni_gas.png", bbox_inches="tight")
    plt.show()

    m_aqn_kg_array = np.linspace(1e-4, 1e-1, 10) * u.kg
    epsilon_array = parameter_variation_t_aqn(quant, m_aqn_kg, frequency_band, "aqn_mass", m_aqn_kg_array, True)

    fig, ax = plot_parameter_variation(r"$m_{aqn}$ [kg]", r"$T_{aqn}$", m_aqn_kg_array, epsilon_array)
    plot_scaling_relation(ax, m_aqn_kg_array, 8/21, np.min(epsilon_array))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(parameter_relations_save_location+"t_aqn_vs_m_aqn.png", bbox_inches="tight")
    plt.show()

    T_gas_array = np.linspace(1e2, 1e6, 100) * u.K
    epsilon_array = parameter_variation_t_aqn(quant, m_aqn_kg, frequency_band, "temp_ion", T_gas_array)

    fig, ax = plot_parameter_variation(r"$T_{gas}$ [K]", r"$T_{aqn}$", T_gas_array, epsilon_array)
    plot_scaling_relation(ax, T_gas_array, 8/7, np.min(epsilon_array))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(parameter_relations_save_location+"t_aqn_vs_t_gas.png", bbox_inches="tight")
    plt.show()

def parameter_variation(quant, m_aqn_kg, frequency_band, parameter_name, parameter_array, mass_variation = False):
    parameter_array_length = len(parameter_array)

    epsilon_array = np.zeros(parameter_array_length) * epsilon_units

    if not mass_variation:
        for i, parameter in enumerate(parameter_array):
            quant[parameter_name] = parameter
            epsilon_array[i] = compute_epsilon_ionized(quant, m_aqn_kg, frequency_band)["aqn_emit"]
    else:
        for i, parameter in enumerate(parameter_array):
            epsilon_array[i] = compute_epsilon_ionized(quant, parameter, frequency_band)["aqn_emit"]

    return epsilon_array

def parameter_variation_t_aqn(quant, m_aqn_kg, frequency_band, parameter_name, parameter_array, mass_variation = False):
    parameter_array_length = len(parameter_array)

    t_aqn_i_array = np.zeros(parameter_array_length) * u.eV

    if not mass_variation:
        for i, parameter in enumerate(parameter_array):
            quant[parameter_name] = parameter
            t_aqn_i_array[i] = compute_epsilon_ionized(quant, m_aqn_kg, frequency_band)["t_aqn_i"]
    else:
        for i, parameter in enumerate(parameter_array):
            t_aqn_i_array[i] = compute_epsilon_ionized(quant, parameter, frequency_band)["t_aqn_i"]

    return t_aqn_i_array

def plot_parameter_variation(parameter1_name, parameter2_name, parameter1_array, parameter2_array):
    fig, ax = plt.subplots(dpi=200)
    ax.plot(parameter1_array, parameter2_array, label="Value")
    ax.set_xlabel(parameter1_name, size=16)
    ax.set_ylabel(parameter2_name, size=16)
    
    return fig, ax 


def plot_scaling_relation(ax, parameter_array, scaling_power, scaling_constant):
    scaled_parameter = parameter_array**scaling_power
    scaled_parameter = scaling_constant * scaled_parameter / scaled_parameter[0] * 10
    ax.plot(parameter_array, scaled_parameter, label=f"Scaling")


def plot_maxwell_boltzmann(v_b=180*u.km/u.s, sigma_v=156*u.km/u.s):
    v = np.linspace(1e-3, 800, 500) * u.km/u.s
    f_v = f_maxbolt(v.value, sigma_v.value, v_b.value)
    plt.plot(v, f_v, label=f'$\sigma_v$={sigma_v.value} km/s, $v_b$={v_b.value} km/s')

    plt.xlabel('Speed $v$ (km/s)')
    plt.ylabel('$f(v)$')
    plt.title('Maxwell-Boltzmann Velocity Distribution with Shift')
    plt.legend()
    plt.grid(True)
    plt.show()

