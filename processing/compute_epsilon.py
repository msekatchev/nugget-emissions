import numpy as np
from astropy import constants as cst
from astropy import units as u

from aqn import *
from constants import *
from survey_parameters import *

sigma_v = 156 * u.km/u.s
v_b = 180 * u.km/u.s
m_aqn_kg = 16.7/1000 * u.kg

quant = load_quant("test_save.pkl")

print(quant)

print(">> loaded data")

print(">> computing epsilon...")
t=tt()
res = compute_epsilon(quant, m_aqn_kg, wavel_min, wavel_max, True, sigma_v, v_b, True)
print(">> done!")
ttt(t,"time taken")

quant["aqn_emit"] = res


save_quant(quant, "test_save.pkl")


