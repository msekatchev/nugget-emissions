# Plot AQN broadband signal for a set of AQN masses

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


quant = {
    'dark_mat': np.array([0.4]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.05]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e5]) * u.K, 
    'dv_ioni':  np.array([0]) * u.km/u.s, 
    'dv_neut':  np.array([0]) * u.km/u.s,
}
enforce_units(quant)

m_aqn_kg_array = np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.9]) * u.kg

n_points = 100

lambda_array = np.linspace(600, 2000, n_points) * u.AA

signal = {}

for i in range(len(m_aqn_kg_array)):

	signal[m_aqn_kg_array[i]] = []

	for j in range(n_points):

		res = compute_epsilon_no_bandwidth(quant, m_aqn_kg_array[i], lambda_array[j], 
	    		True, 156*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi)

		signal[m_aqn_kg_array[i]].append(res.value)

		print(j, end=",")

plt.figure(dpi=300)

for i in range(len(m_aqn_kg_array)):

	plt.plot(lambda_array.value, signal[m_aqn_kg_array[i]], label=str(m_aqn_kg_array[i].value)+" kg")

	print(np.array([signal[m_aqn_kg_array[i]]])/m_aqn_kg_array[i])

plt.legend()
plt.xlabel("Wavelength [AA]", size=15)
plt.ylabel("Signal [photon units]", size=15)
plt.savefig("spectrum.png")
plt.show()

