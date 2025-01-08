import numpy as np
from astropy import constants as cst
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import quad_vec
from scipy.integrate import nquad
import scipy

from aqn import *
from constants import *
from survey_parameters import *

from time import time as tt

m_aqn_kg = 16.7/1000 * u.kg

band_min, band_max = 1300*u.AA, 1700*u.AA

quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4]) * u.K, 
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([0]) * u.km/u.s,
    'xe_frac':  np.array([1])
}
enforce_units(quant)

print("------------------------------------------------------")

print("sigma_v = 50, v_b = 50, Phi = 40.22")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 50*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
print(" --> ", compute_epsilon_no_bandwidth(quant, m_aqn_kg, 1500*u.AA, 
    True, 50*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))

print("sigma_v = 100, v_b = 50, Phi = 7.524")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 100*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 110, v_b = 180, Phi = 1.702")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 110*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 156, v_b = 180, Phi = 1.167")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 156*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 50, v_b = 180, Phi = 0.1312")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 50*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 500, v_b = 180, Phi = 0.06485")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 500*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 110, v_b = 500, Phi = 2.29831e-4")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 110*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 500, v_b = 500, Phi = 0.04198")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 500*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))

print("------------------------------------------------------")


# ------------------------------------------------------
# sigma_v = 50, v_b = 50, Phi = 40.22
#  -->  [40.22376995] ph / (Angstrom s sr cm2)
#  -->  40.24228655597436 cm
# sigma_v = 100, v_b = 50, Phi = 7.524
#  -->  [7.52237539] ph / (Angstrom s sr cm2)
# sigma_v = 110, v_b = 180, Phi = 1.702
#  -->  [1.70101703] ph / (Angstrom s sr cm2)
# sigma_v = 156, v_b = 180, Phi = 1.167
#  -->  [1.16878976] ph / (Angstrom s sr cm2)
# sigma_v = 50, v_b = 180, Phi = 0.1312
#  -->  [0.13118818] ph / (Angstrom s sr cm2)
# sigma_v = 500, v_b = 180, Phi = 0.06485
#  -->  [0.06527633] ph / (Angstrom s sr cm2)
# sigma_v = 110, v_b = 500, Phi = 2.29831e-4
#  -->  [0.00023186] ph / (Angstrom s sr cm2)
# sigma_v = 500, v_b = 500, Phi = 0.04198
#  -->  [0.04067563] ph / (Angstrom s sr cm2)
# ------------------------------------------------------







