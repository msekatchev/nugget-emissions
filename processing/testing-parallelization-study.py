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




sigma_v = 156 * u.km/u.s
v_b = 180 * u.km/u.s

m_aqn_kg = 16.7/1000 * u.kg

N_array = np.logspace(0,3,2)
N_array = np.array([1,5,10])

t_non_parallel = []
t_parallel = []

for i in range(len(N_array)):

	N = int(N_array[i])

	print(N)

	quant = {
	    'dark_mat': np.random.uniform(0.2, 0.4, size=N) * u.GeV / u.cm**3 * GeV_to_g,
	    'ioni_gas': np.random.uniform(0.005, 0.02, size=N) * 1 / u.cm**3,
	    'neut_gas': np.random.uniform(0, 0.01, size=N) * 1 / u.cm**3,
	    'temp_ion': np.random.uniform(1e3, 1e5, size=N) * u.K,
	    'dv_ioni': np.random.uniform(50, 300, size=N) * u.km / u.s,
	    'dv_neut': np.random.uniform(0, 50, size=N) * u.km / u.s,
	}

	enforce_units(quant)


	t=tt()
	res = compute_epsilon(quant, m_aqn_kg, 1300*u.AA, 1700*u.AA, True, sigma_v, v_b, False)
	t_non_parallel.append(tt()-t)
	ttt(t,"1")

	t=tt()
	res = compute_epsilon(quant, m_aqn_kg, 1300*u.AA, 1700*u.AA, True, sigma_v, v_b, True)
	t_parallel.append(tt()-t)
	ttt(t,"2")

plt.figure(dpi=200)
plt.plot(N_array, t_non_parallel, "-", color="blue", label="Single core")
plt.plot(N_array, t_parallel, "-", color="green", label="Parallelized, 8 cores")
plt.xscale("log")
plt.legend()
plt.show()








