import sys

import numpy as np
from astropy import constants as cst
from astropy import units as u

from aqn import *
from constants import *
from survey_parameters import *

from multiprocessing import cpu_count

print(cpu_count())


m_aqn_kg = 0.1 * u.kg

quant = load_quant("../data/cubes.pkl")

print(len(quant["ioni_gas"]))


quant = {
    'dark_mat': np.array([0.3,1,1,1,1,1]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01,1,1,1,1,1]) * 1/u.cm**3,
    'ioni_gas_c': np.array([0.01,1,1,1,1,1]) * 1/u.cm**3,
    'neut_gas': np.array([0,1,1,1,1,1]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4,1,1,1,1,1]) * u.K, 
    'dv':  np.array([0,1,1,1,1,1]) * u.km/u.s
}

L = 100
quant = {
    'dark_mat': np.linspace(0.1,1,L) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.linspace(0.1,1,L) * 1/u.cm**3,
    'ioni_gas_c': np.linspace(0.1,1,L) * 1/u.cm**3,
    'neut_gas': np.linspace(0.1,1,L) * 1/u.cm**3, 
    'temp_ion': np.linspace(0.1,1,L) * u.K, 
    'dv':  np.linspace(0.1,1,L) * u.km/u.s
}


enforce_units(quant)



# modify dv to be unitless
quant["dv_ioni"] = quant["dv"] / cst.c.to(u.km/u.s)
quant["ioni_gas"] = quant["ioni_gas_c"]

print(">> loaded data")




# print(">> computing epsilon...")
t=tt()
res = compute_epsilon(quant, m_aqn_kg, wavel_min, wavel_max, True, sigma_v=1*u.km/u.s, v_b=1*u.km/u.s, use_fire_dv=True, 
    parallel=True)
print(">> done!")
ttt(t,"time taken")

# quant["aqn_emit"] = res

# print(quant["aqn_emit"])




































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


