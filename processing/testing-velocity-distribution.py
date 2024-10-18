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


###############################################################################
# Testing with Xunyu's values

n_bar = 0.01 * 1/u.cm**3
Dv = 220 * u.km/u.s
f = 1
g = 0.1
T_p = 10**4 * u.K
R = calc_R_AQN((16.7 * u.g).to(u.kg))

print(f"n_bar={n_bar}\nDv={Dv}\nf={f}\ng={g}\nT_p={T_p}\nR={R}")

print(">>\t",T_AQN_ionized2(
    n_bar = 0.01 * 1/u.cm**3,
    Dv = 220 * u.km/u.s / cst.c.to(u.km/u.s),
    f = 1,
    g = 0.1,
    T_p = 10**4 * u.K * K_to_eV,
    R = calc_R_AQN((16.7 * u.g).to(u.kg))))

print("\n\n")

n_bar = 0.01 * 1/u.cm**3
Dv = 220 * u.km/u.s
f = 1
g = 0.1
T_p = 1.5*10**5 * u.K
R = calc_R_AQN((16.7 * u.g).to(u.kg))

print(f"n_bar={n_bar}\nDv={Dv}\nf={f}\ng={g}\nT_p={T_p}\nR={R}")

print(">>\t",T_AQN_ionized2(
    n_bar = 0.01 * 1/u.cm**3,
    Dv = 220 * u.km/u.s / cst.c.to(u.km/u.s),
    f = 1,
    g = 0.1,
    T_p = 1.5*10**5 * u.K * K_to_eV,
    R = calc_R_AQN((16.7 * u.g).to(u.kg))))

###############################################################################

###############################################################################
# Investigation of T_AQN VS dv, ioni_gas, m_aqn and T_gas_eff
t_aqn_parameter_relations_study(quant.copy(), m_aqn_kg, frequency_band)





# T_AQN_array = np.zeros(len(velocity_array)) * u.K
# for i, velocity in enumerate(velocity_array):
#     quant["dv_ioni"] = velocity
#     enforce_units(quant)
#     T_AQN_array[i] = compute_epsilon_ionized(quant, m_aqn_kg, frequency_band)["t_aqn_i"] / K_to_eV


# fig, ax = plot_parameter_variation(r"$\Delta v$ [km/s]", r"$T_{aqn}$", velocity_array, T_AQN_array)
# # plot_scaling_relation(ax, velocity_array, 20/7, np.max(T_AQN_array))
# ax.plot(velocity_array, np.min(T_AQN_array) * (velocity_array.value * (1/(1+velocity_array.value**2)**2))**(4/7), label="Scaling")
# ax.set_xscale("log")
# ax.set_yscale("log")
# plt.title(r"T_AQN after dv modification, scaling $(dv/(1+dv^2)^2)^{(4/7)}$")
# plt.legend()
# # plt.savefig(parameter_relations_save_location+"t_aqn_vs_dv.png", bbox_inches="tight")
# plt.show()

























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
