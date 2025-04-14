import sys

import numpy as np
from astropy import constants as cst
from astropy import units as u

from aqn import *
from constants import *
from survey_parameters import *

from multiprocessing import cpu_count

import matplotlib.pyplot as plt

print(cpu_count())

'''
def compute_epsilon(quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b, use_fire_dv=False, parallel=False):

    if parallel:
        
        cpu_num = cpu_count()
        batches = split_quant(quant, cpu_num)

        if use_fire_dv:
            batch_results = Parallel(n_jobs=cpu_num)(
                delayed(process_batch_FIRE_dv)(batch, m_aqn_kg, band_min, band_max, adjust_T_gas)
                for batch in batches)
        else:
            batch_results = Parallel(n_jobs=cpu_num)(
                delayed(process_batch)(batch, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b)
                for batch in batches)

        # concatenate non-empty arrays
        return np.concatenate([batch for batch in batch_results if len(batch) > 0], axis=0)

    else:
        if use_fire_dv:
            return process_batch_FIRE_dv(quant, m_aqn_kg, band_min, band_max, adjust_T_gas)
        else:
            return process_batch(quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b)

def simple_integrator(func, a, b, resolution, 
    cube, sigma_v, v_b, m_aqn_kg, band_min, band_max, adjust_T_gas):
    """
    Simple numerical integrator using the trapezoidal rule.

    Parameters:
    func (callable): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    resolution (int): The number of points to use in the integration.

    Returns:
    float: The approximate integral of the function from a to b.
    """
    # Generate linearly spaced points between a and b
    x = np.linspace(a, b, resolution)
    
    # Evaluate the function at each point
    y = func(x, cube, sigma_v, v_b, m_aqn_kg, band_min, band_max, adjust_T_gas)
    
    # Use the trapezoidal rule to compute the integral
    dx = (b - a) / (resolution - 1)
    integral = np.sum((y[:-1] + y[1:]) * dx / 2)
    
    return integral

def process_batch(quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b):
    # print("hiiiii")
    # print(quant["temp_ion"])
    quant_count = len(quant["dark_mat"])
    
    if quant_count == 0:
        
        return []
        
    else:
        batches = split_quant(quant, quant_count)
        
        batch_results = []

        for i in range(quant_count):
            # choose:

            # 1. simple trapezoid integrator
            # batch_results.append(simple_integrator(
            #     epsilon_velocity_integrand, 0.1, 1*(sigma_v.value+v_b.value), 10000,
            #     batches[i], sigma_v.value, v_b.value, m_aqn_kg, band_min, band_max, adjust_T_gas))

            # 2. built-in scipy.quad()
            t = tt()
            batch_results.append(quad(epsilon_velocity_integrand, 0.1, 1*(sigma_v.value+v_b.value), 
                args=(batches[i], sigma_v.value, v_b.value, m_aqn_kg, band_min, band_max, adjust_T_gas),
                epsabs=1e-8, epsrel=1e-8)[0])
            ttt(t, "quad call")
        return batch_results * epsilon_units/u.sr
'''
#--------------------------------------------------------------------------------

# Goal: Determine what is slowing down <Phi> computation

# initialize m_AQN
m_aqn_kg = 0.0167 * u.kg

# pick a quant to work with

# 1. load in the actual quant from file
quant = load_quant("../data/filtered-location-voxels/april-9-2025/cubes.pkl")
# print(quant.keys())
quant["dv_ioni"] = (quant["dv"] / cst.c).to(u.dimensionless_unscaled)


# 2. quant for a single voxel
# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e4]) * u.eV, 
#     'dv_ioni':  np.array([220]) * u.km/u.s, 
#     'dv_neut':  np.array([0]) * u.km/u.s,
# }

# quant["dv_ioni"] = (quant["dv_ioni"] / cst.c).to(u.dimensionless_unscaled)

# 3. quant for 2 voxels
# quant = {
#     'dark_mat': np.array([0.3,0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01,0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0,0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e4,1e4]) * u.K, 
#     'dv_ioni':  np.array([220,220]) * u.km/u.s, 
#     'dv_neut':  np.array([0,0]) * u.km/u.s,
# }

# 3. quant for 100 voxels
# L = 10
# quant = {
#     'dark_mat': np.linspace(0.1,1,L) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.linspace(0.1,1,L) * 1/u.cm**3,
#     'ioni_gas_c': np.linspace(0.1,1,L) * 1/u.cm**3,
#     'neut_gas': np.linspace(0.1,1,L) * 1/u.cm**3, 
#     'temp_ion': np.linspace(0.1,1,L) * u.K, 
#     'dv':  np.linspace(0.1,1,L) * u.km/u.s
# }

# fix units
# enforce_units(quant)

# unnecessary if working with MB dv:
# modify dv to be unitless
# quant["dv_ioni"] = quant["dv"] / cst.c.to(u.km/u.s)
# quant["ioni_gas"] = quant["ioni_gas_c"]

# pick a v_b and sigma_v for MB dv distribution
v_b = 200 * u.km/u.s
sigma_v = v_b / np.sqrt(2) # 20 * u.km/u.s

# compute epsilon
# res = compute_epsilon(quant, m_aqn_kg, wavel_min, wavel_max, 
#                       adjust_T_gas=True, sigma_v=sigma_v, v_b=v_b, 
#                       use_fire_dv=False, parallel=False)
# simple 0.6 kpc estimate
# res = res*(0.6*u.kpc).to(u.cm)/(4*np.pi)
# print(res)


# compute epsilon and compare w/ Xunyu's estimate
print("sigma_v = 156, v_b = 180, Phi = 1.167")
print(" --> ", compute_epsilon(quant, m_aqn_kg, wavel_min, wavel_max, 
    True, 156*u.km/u.s, 180*u.km/u.s, parallel=False, use_fire_dv=True)*(0.6*u.kpc).to(u.cm)/(4*np.pi))





































# plot MB distribution
# plt.figure(dpi=300)
# # plt.hist(cubes["dv"][full_density_filter], bins=30, alpha = 0.4, color="black", density=True, edgecolor="black", label="FIRE filtered voxels")
# dv_array = np.linspace(1,1000,1000)
# v0 = 220
# maxwell = f_maxbolt(dv_array, sigma_v = sigma_v.value, v_b = v_b.value)
# plt.plot(dv_array, maxwell, color="blue", label=r"MB, v$_0$=220, $\sigma_{\rm{v}}$=156")
# plt.xlabel(r"$\Delta\rm{v}$ [km/s]", size=15)
# plt.ylabel(r"$\rm{P}(\Delta\rm{v})$", size=15)
# plt.legend(fontsize=15)
# # plt.savefig("../visuals/dv-maxwell-vs-fire.png", bbox_inches = "tight")
# # plt.savefig("../visuals/dv-maxwell-vs-fire.pdf", bbox_inches = "tight")
# plt.show()




































# sigma_v = 156 * u.km/u.s
# v_b = 180 * u.km/u.s

# band_min, band_max = wavel_min, wavel_max

# m_aqn_kg = 16.7/1000 * u.kg

# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e4]) * u.K, 
#     'dv_ioni':  np.array([200]) * u.km/u.s, 
#     'dv':  np.array([0]) * u.km/u.s,
# }
# enforce_units(quant)

# m_aqn_kg = 16.7/1000 * u.kg

# print("------------------------------------------------------")

# print(quant)
# quant["dv_ioni"] = quant["dv_ioni"] / cst.c.to(u.km/u.s)

# print("sigma_v = 100, v_b = 50, Phi = 7.524")
# print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
#     True, sigma_v=10*u.km/u.s, v_b=200*u.km/u.s, use_fire_dv=False, parallel=False)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
#     True, sigma_v=1, v_b=1, use_fire_dv=True, parallel=False)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


# # compute_epsilon(quant, m_aqn_kg, wavel_min, wavel_max, True, sigma_v=1, v_b=1, use_fire_dv=True, parallel=True)
























# # sample usage:
# # python3 compute_epsilon.py 0.16 test_save.pkl
# m_aqn_kg = float(sys.argv[1]) * u.kg
# load_name = sys.argv[2]
# save_name = sys.argv[3]

# print(m_aqn_kg)
# print(load_name)
# print(save_name)

# # m_aqn_kg = 16.7/1000 * u.kg

# quant = load_quant("../data/filtered-location-voxels/"+load_name+".pkl")

# # modify dv to be unitless
# quant["dv"] = quant["dv"] / cst.c.to(u.km/u.s)

# print(">> loaded data")

# quant["aqn_emit"] = quant["dark_mat"].copy() / quant["dark_mat"] * u.K
# save_quant(quant, "../data/filtered-location-voxels/"+save_name+"-in-progress.pkl")

# print(">> successfully tested save location")

# print(">> computing epsilon...")
# t=tt()
# res = compute_epsilon(quant, m_aqn_kg, wavel_min, wavel_max, True, sigma_v=1, v_b=1, use_fire_dv=True, parallel=True)
# print(">> done!")
# ttt(t,"time taken")

# quant["aqn_emit"] = res


# save_quant(quant, "../data/filtered-location-voxels/"+save_name+".pkl")


