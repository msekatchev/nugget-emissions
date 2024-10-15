import matplotlib.pyplot as plt
import matplotlib

from aqn import *
from constants import *
from notebook_functions import *
from survey_parameters import *

# specify frequency resolution and create frequency band array
# Change frequency range within survey_parameters.py
dnu = 1e14*u.Hz # 1e9 for WMAP, 1e14 for GALEX
frequency_band = np.arange(f_min_hz.value, f_max_hz.value, dnu.value) * u.Hz
nu_range = f_max_hz - f_min_hz

print(frequency_band[1] - frequency_band[0], dnu)

quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.014]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4]) * u.K, 
    'dv_ioni':  np.array([200]) * u.km/u.s, 
    'dv_neut':  np.array([200]) * u.km/u.s,
}
enforce_units(quant)

m_aqn_kg = 0.01 * u.kg

epsilon_parameter_relations_study(quant, m_aqn_kg, frequency_band)
