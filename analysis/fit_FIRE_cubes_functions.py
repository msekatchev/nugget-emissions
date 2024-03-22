import gizmo_analysis as gizmo  # rename these packages for brevity
import utilities as ut  # rename these packages for brevity

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from astropy import units as u
from astropy import constants as cst

# from aqn import *
# from constants import *
import pickle
import scipy.stats as stats

from scipy.stats import bootstrap

# define cube resolution parameters
# cube_length       = 20   # kpc
# voxel_resolution  = 2**9 # bins / kpc
# print(f">> voxels per length: {voxel_resolution}")

# voxel_length = cube_length / voxel_resolution * u.kpc
# voxel_volume = voxel_length**3

# create the plot directly from binning masses
def generate_rho_distrib_parts(comp, distance_bins):
    # pick out particles within the max distance of centre
    within_mw = part[comp].prop('host.distance.total') <= max_distance_kpc.value
    
    # array of distances of each particle to the host Galaxy centre
    dm = part[comp].prop('host.distance.total')[within_mw]
    
    # All DM particles have the same mass:
    mass_comp = (part[comp]['mass'][within_mw][0] * u.Msun).to(u.kg)

    res = np.histogram(dm, bins=distance_bins.value)
    N, R = res[0], res[1] * u.kpc
    R_inner, R_outer = R[0:-1], R[1:]

    N_error = np.sqrt(N)

    m       = N * mass_comp
    m_error = m * N_error / N

    V_outer = 4/3 * np.pi * (R_outer).to(u.m)**3
    V_inner = 4/3 * np.pi * (R_inner).to(u.m)**3

    rho       = m / (V_outer-V_inner)
    rho_error = rho * m_error / m
    
    return rho, rho_error

##############################################################################################
def generate_rho_distrib_cubes(cube, distance_bins, distances, voxel_resolution):
    density = cube.reshape((voxel_resolution**3, 1)).T[0]

    rho, bin_edges, bin_numbers = stats.binned_statistic(distances, 
                      density, 
                      statistic='sum', 
                      bins=distance_bins)

    voxel_length = 20 / voxel_resolution * u.kpc
    voxel_volume = voxel_length**3
    
    rho = rho * u.kg/u.m**3
    R = bin_edges * u.kpc
    R_inner, R_outer = R[0:-1], R[1:]
    
    V_outer = 4/3 * np.pi * (R_outer).to(u.m)**3
    V_inner = 4/3 * np.pi * (R_inner).to(u.m)**3
    
    rho       = rho * voxel_volume.to(u.m**3) / (V_outer-V_inner)

    return rho

##############################################################################################
def plot_cubes_parts(quant, quant_cube, quant_part, distance_bins, quant_errors):
    plt.figure(dpi=300)
    plt.errorbar(distance_bins[1:], binned_cubes[quant], yerr=quant_errors, marker="o", markersize=3, linestyle = 'none',color='blue', ecolor="gray", elinewidth=0.7, label="Binned Cube")
    plt.errorbar(distance_bins[1:], binned_parts[quant], yerr=quant_errors, marker="^", markersize=3, linestyle = 'none',color='green', ecolor="gray", elinewidth=0.7, label="Binned Particles")
    # plt.bar(distance_bins[0:len(distance_bins) - 1].value, binned_dm_density_si.value, width=kpc_per_bin ,color="burlywood", alpha = 0.75)
    # plt.errorbar(R_outer[R_outer.value < 1], rho[R_outer.value < 1], yerr=rho_error[R_outer.value < 1], marker=".", markersize=1, linestyle = 'none',color='red', ecolor="gray", elinewidth=0.7)
    plt.title(quant)
    plt.xlabel("R [kpc]", size=20)
    plt.ylabel(r'$\rho$'+"  [kg$\cdot$m$^{-3}$]", size = 20)
    # plt.scatter(A, B, s=3)
    plt.legend(fontsize=20)
    
    plt.savefig("../visuals/"+quant+"-binned-particles-and-cubes.png", bbox_inches='tight')
    plt.savefig("../visuals/"+quant+"-binned-particles-and-cubes.png", bbox_inches='tight')

    plt.show()
    
    plt.figure(dpi=300)
    plt.errorbar(distance_bins[1:], binned_cubes[quant]- binned_parts[quant], yerr=quant_errors, marker="o", markersize=3, linestyle = 'none',color='red', ecolor="gray", elinewidth=0.7)
    plt.title(quant + ": binned cube - binned particle densities")
    plt.xlabel("R [kpc]", size=20)
    plt.ylabel(r'$\rho$'+"  [kg$\cdot$m$^{-3}$]", size = 20)

    plt.savefig("../visuals/"+quant+"-binned-particles-vs-cubes.png", bbox_inches='tight')
    plt.savefig("../visuals/"+quant+"-binned-particles-vs-cubes.png", bbox_inches='tight')
    
    plt.show()

def plot_cubes_parts_cum(quant, quant_cube, quant_part, distance_bins, quant_errors):
    plt.figure(dpi=300)
    plt.errorbar(distance_bins[1:], binned_cubes[quant], yerr=quant_errors, marker="o", markersize=3, linestyle = 'none',color='blue', ecolor="gray", elinewidth=0.7, label="Binned Cube")
    plt.errorbar(distance_bins[1:], binned_parts[quant], yerr=quant_errors, marker="^", markersize=3, linestyle = 'none',color='green', ecolor="gray", elinewidth=0.7, label="Binned Particles")
    # plt.bar(distance_bins[0:len(distance_bins) - 1].value, binned_dm_density_si.value, width=kpc_per_bin ,color="burlywood", alpha = 0.75)
    # plt.errorbar(R_outer[R_outer.value < 1], rho[R_outer.value < 1], yerr=rho_error[R_outer.value < 1], marker=".", markersize=1, linestyle = 'none',color='red', ecolor="gray", elinewidth=0.7)
    plt.title(quant)
    plt.xlabel("R [kpc]", size=20)
    plt.ylabel(r'$m$'+"  [kg]", size = 20)
    # plt.scatter(A, B, s=3)
    plt.legend(fontsize=20)
    
    plt.savefig("../visuals/"+quant+"-binned-particles-and-cubes-cum.png", bbox_inches='tight')
    plt.savefig("../visuals/"+quant+"-binned-particles-and-cubes-cum.png", bbox_inches='tight')

    plt.show()
    
    plt.figure(dpi=300)
    plt.errorbar(distance_bins[1:], binned_cubes[quant]- binned_parts[quant], yerr=quant_errors, marker="o", markersize=3, linestyle = 'none',color='red', ecolor="gray", elinewidth=0.7)
    plt.title(quant + ": binned cube - binned particle masses")
    plt.xlabel("R [kpc]", size=20)
    plt.ylabel(r'$m$'+"  [kg]", size = 20)

    plt.savefig("../visuals/"+quant+"-binned-particles-vs-cubes-cum.png", bbox_inches='tight')
    plt.savefig("../visuals/"+quant+"-binned-particles-vs-cubes-cum.png", bbox_inches='tight')
    
    plt.show()

def plot_covariance_matrix(cov_matrix, mass, distance_bins, sampled_densities, name):    
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300)
    
   
    # Plot for the first column
    ax = axes[0]
    ax.set_title(f"True $\\rho$ - Mean Sampled $\\rho$")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")
    
    mass_diff = mass.value - np.mean(sampled_densities, axis=0)

    ax.scatter(distance_bins[1:], mass_diff)
    ax.axhline(0, color="r", linestyle="--")
    for i in range(len(distance_bins) - 1):
        ax.plot([distance_bins[i+1].value, distance_bins[i+1].value], [0, mass_diff[i]], color='blue')

    ax = axes[1]
    ax.set_title(f"Correlations")
    
    x, y = np.meshgrid(distance_bins.value, distance_bins.value)
    res = ax.pcolor(x, y, np.corrcoef(np.transpose(sampled_densities)))

    ax.invert_yaxis()
    plt.colorbar(res)

    
    # Plot for the second column
    ax = axes[2]
    ax.set_title(f"inv($\Sigma$)$\cdot\Sigma-1$")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Distance")

    x, y = np.meshgrid(distance_bins.value, distance_bins.value)
    res = ax.pcolor(x, y, np.linalg.inv(cov_matrix) @ cov_matrix - np.diag(np.ones(len(distance_bins)-1)))
    ax.invert_yaxis()
    plt.colorbar(res)    
    
    plt.tight_layout()   
    
    plt.savefig("../visuals/cov-degeneracy-result-"+name+".png", bbox_inches='tight')
    plt.savefig("../visuals/cov-degeneracy-result-"+name+".svg", bbox_inches='tight')
    plt.show()
##############################################################################################
def generate_rho_distrib_cubes_cum(cube, distance_bins, distances, voxel_resolution):
    
    density = cube.reshape((voxel_resolution**3, 1)).T[0]
    
    density_cum = np.array([np.sum(density[distances < distance_bins[i]]).value * voxel_volume.to(u.m**3).value for i in range(1,len(distance_bins))]) * u.kg
    # / (4/3 * np.pi * distance_bins[i]**3).value
    return density_cum
##############################################################################################
    # create the plot directly from binning masses
def generate_rho_distrib_parts_cum(comp, distance_bins):
    # pick out particles within the max distance of centre
    within_mw = part[comp].prop('host.distance.total') <= max_distance_kpc.value
    
    # array of distances of each particle to the host Galaxy centre
    dm = part[comp].prop('host.distance.total')[within_mw] * u.kpc

    mass_comp = (part[comp]['mass'][within_mw] * u.Msun)
    
    mass_cum = np.array([(np.sum(mass_comp[dm < distance_bins[i]])).to(u.kg).value for i in range(1,
              len(distance_bins))]) * u.kg

    N = np.array([np.sum(dm < distance_bins[i]) for i in range(1, len(distance_bins))])
    
    mass_cum_error = mass_cum / np.sqrt(N)
    
    return mass_cum, mass_cum_error
##############################################################################################
    # create the plot directly from binning masses
def generate_rho_distrib_parts_cum(data, distance_bins):
    masses, distances = data[0], data[1]
    # print(masses)
    masses_cum = np.array([(np.sum(masses[distances < distance_bins[i]])).to(u.kg).value for i in range(1,
                  len(distance_bins))]) * u.kg
    
    return masses_cum #/ (4/3 * np.pi * distance_bins[1:]**3).to(u.m**3)
##############################################################################################
# create the plot directly from binning masses
def generate_rho_distrib_parts(data, distance_bins):
    masses, distances = data[0], data[1]
    
    # All DM particles have the same mass:
    mass_comp = (masses[0] * u.Msun).to(u.kg)  ## !!! Update this!

    res = np.histogram(distances, bins=distance_bins.value)
    N, R = res[0], res[1] * u.kpc
    R_inner, R_outer = R[0:-1], R[1:]

    N_error = np.sqrt(N)

    m       = N * mass_comp

    V_outer = 4/3 * np.pi * (R_outer).to(u.m)**3
    V_inner = 4/3 * np.pi * (R_inner).to(u.m)**3

    rho       = m / (V_outer-V_inner)
    
    return rho
##############################################################################################
def my_bootstrap(data, distance_bins, bsfunction, fraction, num_samples):
    length = len(data[0])
    samples_indexes = np.random.choice(length, size=(num_samples, round(fraction * length)))
    
    res = np.array([bsfunction([data[0][samples_indexes[i,:]], data[1][samples_indexes[i,:]]], distance_bins) for i in range(num_samples)])
    
    return res*1/fraction, (1/fraction)**2 * np.cov(res.T)
#                ^^^ Need to find source
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################