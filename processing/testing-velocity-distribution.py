import matplotlib.pyplot as plt
import matplotlib

from aqn import *
from constants import *
from notebook_functions import *
from survey_parameters import *

from scipy.integrate import quad

# specify frequency resolution and create frequency band array
# Change frequency range within survey_parameters.py
dnu = 1e14*u.Hz # 1e9 for WMAP, 1e14 for GALEX
frequency_band = np.arange(f_min_hz.value, f_max_hz.value, dnu.value) * u.Hz
nu_range = f_max_hz - f_min_hz

quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.014]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4]) * u.K, 
    'dv_ioni':  np.array([250]) * u.km/u.s, 
    'dv_neut':  np.array([200]) * u.km/u.s,
}
enforce_units(quant)

m_aqn_kg = 0.01 * u.kg

sigma_v, v_b = 156, 180

# epsilon_parameter_relations_study(quant, m_aqn_kg, frequency_band)

# 
###############################################################################
# Testing with Xunyu's values
quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e6]) * u.K, 
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([200]) * u.km/u.s,
}

m_aqn_kg = 16.7/1000 * u.kg



# n_bar = 0.01 * 1/u.cm**3
# Dv = 220 * u.km/u.s
# f = 1
# g = 0.1
# T_p = 10**4 * u.K
# R = calc_R_AQN((16.7 * u.g).to(u.kg))

# print(f"n_bar={n_bar}\nDv={Dv}\nf={f}\ng={g}\nT_p={T_p}\nR={R}")
# print(f"c={cst.c.to(u.km/u.s)}")

# print(">>\t",T_AQN_ionized2(
#     n_bar = 0.01 * 1/u.cm**3,
#     Dv = 220 * u.km/u.s / cst.c.to(u.km/u.s),
#     f = 1,
#     g = 0.1,
#     T_p = 10**4 * u.K * K_to_eV,
#     R = calc_R_AQN((16.7 * u.g).to(u.kg))))


# print("\n\n")

# n_bar = 0.01 * 1/u.cm**3
# Dv = 220 * u.km/u.s
# f = 1
# g = 0.1
# T_p = 1.5*10**5 * u.K
# R = calc_R_AQN((16.7 * u.g).to(u.kg))

# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1.5e5]) * u.K, 
#     'dv_ioni':  np.array([220]) * u.km/u.s, 
#     'dv_neut':  np.array([200]) * u.km/u.s,
# }

# enforce_units(quant)

# print(f"n_bar={n_bar}\nDv={Dv}\nf={f}\ng={g}\nT_p={T_p}\nR={R}")

# print(">>\t",T_AQN_ionized2(
#     n_bar = 0.01 * 1/u.cm**3,
#     Dv = 220 * u.km/u.s / cst.c.to(u.km/u.s),
#     f = 1,
#     g = 0.1,
#     T_p = 1.5*10**5 * u.K * K_to_eV,
#     R = calc_R_AQN((16.7 * u.g).to(u.kg))))

# print("----->", compute_epsilon_ionized(quant.copy(), m_aqn_kg, frequency_band)["aqn_emit"] * (0.6*u.kpc).to(u.cm)/(4*np.pi))

# print("\n\n")

# print("Some more constants:")
# print(f"c={cst.c.to(u.km/u.s)}")
# print(f"alpha={cst.alpha}")
# print(f"m_p={m_p_erg}")

###############################################################################
# import matplotlib
# matplotlib.use('TkAgg')
###############################################################################


# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e4]) * u.K, 
#     'dv_ioni':  np.array([220]) * u.km/u.s, 
#     'dv_neut':  np.array([200]) * u.km/u.s,
# }

quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4]) * u.K, 
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([200]) * u.km/u.s,
}

enforce_units(quant)
print("----->", compute_epsilon_ionized(quant.copy(), m_aqn_kg, frequency_band)["aqn_emit"] * (0.6*u.kpc).to(u.cm)/(4*np.pi))

# print("Initial temp_ion is::::")
# print(quant["temp_ion"])
# print("-------------------")

# Investigation of T_AQN VS dv, ioni_gas, m_aqn and T_gas_eff
# t_aqn_parameter_relations_study(quant.copy(), m_aqn_kg, frequency_band)

# Investigation of epsilon VS dv, ioni_gas, m_aqn and T_gas_eff
# epsilon_parameter_relations_study(quant.copy(), m_aqn_kg, frequency_band)


T_AQN = T_AQN_ionized2(n_bar=quant["ioni_gas"], 
    Dv=quant["dv_ioni"], 
    f=1, 
    g=0.1, 
    T_p=quant["temp_ion"], 
    R=calc_R_AQN(m_aqn_kg))

# print(T_AQN)
from astropy import constants as cst
from astropy import units as u

T_AQN = 102.1*u.eV
w = 1500 * u.AA 
nu = w.to(u.Hz, equivalencies=u.spectral()) # 1.999e15 [Hz]
L = (0.6*u.kpc).to(u.m)
R = calc_R_AQN(m_aqn_kg).to(u.m) # 2.25 [m]
n_AQN = 2/5 * quant["dark_mat"] / m_aqn_kg # 1.28e-20 [1/m**3]
X = T_AQN.to(u.J)
C = 8 * cst.alpha**(5/2) / (45*cst.hbar**2*cst.c**2)
F = C * X**3 * (X/cst.m_e/cst.c**2)**(1/4) * H(2*np.pi*cst.hbar*nu/X)
Phi = L*R**2*n_AQN*F/(2*np.pi*cst.hbar*w) * (1*u.m/(100*u.cm))**2
print(Phi)


# J_to_eV = (1*u.J).to(u.eV) / u.J
# m2_to_cm2 = (1*u.m**2).to(u.cm**2) / u.m**2
# A = 8 * cst.alpha**(5/2) / (45*cst.hbar**2*cst.c**2) * 1/J_to_eV**2

# B = T_AQN**3

# C = (T_AQN/m_e_eV)**(1/4)

# nu = np.mean(frequency_band)

# D = H(2*np.pi*cst.hbar*nu / T_AQN * J_to_eV)

# F = A*B*C*D*eV_to_erg/m2_to_cm2/u.Hz

# print(F)

# w = nu.to(u.AA, equivalencies=u.spectral())
# C = (erg_hz_cm2).to(photon_units*u.sr, u.spectral_density(w))

# F = F * C / erg_hz_cm2 * 2*np.pi

# L = (0.6*u.kpc).to(u.cm)
# R = calc_R_AQN(m_aqn_kg)

# print(F*L*R**2*quant["dark_mat"]/m_aqn_kg/(2*np.pi*cst.hbar*J_to_eV*w.to(u.cm)*eV_to_erg/inverg_to_cm)*(1/100*u.m/u.cm)**3)

# print(quant["dark_mat"])

# print(D)




















# plot_maxwell_boltzmann()

def compute_epsilon_integrand(v, quant, sigma_v, v_b, m_aqn_kg, frequency_band):
    f_res = f_maxbolt(v, sigma_v, v_b)
    quant_copy = quant.copy()
    quant_copy["dv_ioni"] = (v*u.km/u.s) / cst.c.to(u.km/u.s)

    e_res = compute_epsilon_ionized(quant_copy, m_aqn_kg, frequency_band)["aqn_emit"].value

    return e_res * f_res

# print(compute_epsilon_integrand(0.5, quant, 156, 180, m_aqn_kg, frequency_band))


def compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, sigma_v, v_b):
    # result, error = quad(f_maxbolt, 0, 10000, args=(sigma_v.value, v_b.value))
    result, error = quad(compute_epsilon_integrand, 0.1, 800, 
        args=(quant, sigma_v.value, v_b.value, m_aqn_kg, frequency_band))

    print(f"Integral result: {result:.4e}, Estimated error: {error:.4e}")

    return result * epsilon_units

# print(compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 156*u.km/u.s, 180*u.km/u.s))
# print(compute_epsilon_ionized(quant, m_aqn_kg, frequency_band)["aqn_emit"][0])

def parameter_variation_integrand(quant, sigma_v, v_b, m_aqn_kg, frequency_band, parameter_name, parameter_array, mass_variation = False):
    parameter_array_length = len(parameter_array)

    epsilon_array = np.zeros(parameter_array_length)

    if not mass_variation:
        for i, parameter in enumerate(parameter_array):
            quant[parameter_name] = parameter
            epsilon_array[i] = compute_epsilon_integrand(quant["dv_ioni"]*cst.c.to(u.km/u.s).value, quant, sigma_v, v_b, m_aqn_kg, frequency_band)[0]
    else:
        for i, parameter in enumerate(parameter_array):
            epsilon_array[i] = compute_epsilon_integrand(quant["dv_ioni"]*cst.c.to(u.km/u.s).value, quant, sigma_v, v_b, parameter, frequency_band)[0]

    return epsilon_array * epsilon_units


# parameter_relations_save_location = "../visuals/parameter_relations/"

# velocity_array = np.linspace(1e-5, 1e-3, 100)
# epsilon_array = parameter_variation_integrand(quant, sigma_v, v_b, m_aqn_kg, frequency_band, "dv_ioni", velocity_array)
# fig, ax = plot_parameter_variation(r"$\Delta v$", r"$\epsilon$", velocity_array, epsilon_array)
# plot_scaling_relation(ax, velocity_array, 2, np.min(epsilon_array))
# ax.set_xscale("log")
# ax.set_yscale("log")
# plt.legend()
# plt.savefig(parameter_relations_save_location+"epsilon_int_vs_dv.png", bbox_inches="tight")
# plt.show()

# print((22 * u.km/u.s )/cst.c.to(u.km/u.s) )
# print(1e-3 * cst.c.to(u.km/u.s))
