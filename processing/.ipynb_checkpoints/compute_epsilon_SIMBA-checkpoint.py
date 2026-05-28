import sys

import numpy as np
from astropy import constants as cst
from astropy import units as u

from aqn import *
from constants import *
from survey_parameters import *

# sample usage:
# python3 compute_epsilon.py 0.16 test_save.pkl
# python3 compute_epsilon_FIRE_dv.py 0.5 april-12-2025/cubes april-12-2025/m-0_5kg-FIRE-dv
# python3 compute_epsilon_FIRE_dv.py 0.1 april-12-2025/cubes april-12-2025/m-0_1kg-FIRE-dv
# python3 compute_epsilon_FIRE_dv.py 0.01 april-12-2025/cubes april-12-2025/m-0_01kg-FIRE-dv

m_aqn_kg = float(sys.argv[1]) * u.kg
load_name = sys.argv[2]
save_name = sys.argv[3]

print(m_aqn_kg)
print(load_name)
print(save_name)

# m_aqn_kg = 16.7/1000 * u.kg

quant = load_quant("../data/filtered-location-voxels/"+load_name+".pkl")

# print(quant["dv"])
# modify dv to be unitless
# quant["dv_ioni"] = quant["dv"] / cst.c.to(u.km/u.s)
quant["dv_ioni"] = (quant["dv"] / cst.c).to(u.dimensionless_unscaled)
# quant["ioni_gas"] = quant["ioni_gas_c"]
# enforce_units(quant)
print(">> loaded data")

quant["aqn_emit"] = quant["dark_mat"].copy() / quant["dark_mat"] * u.K
save_quant(quant, "../data/filtered-location-voxels/"+save_name+"-in-progress.pkl")

print(">> successfully tested save location")

print(">> computing epsilon...")
t=tt()
res = compute_epsilon(quant, m_aqn_kg, wavel_min, wavel_max, True, sigma_v=1, v_b=1, use_fire_dv=True, parallel=True)
print(">> done!")
ttt(t,"time taken")

quant["aqn_emit"] = res


save_quant(quant, "../data/filtered-location-voxels/"+save_name+".pkl")


