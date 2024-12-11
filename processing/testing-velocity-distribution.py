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
# frequency_band = np.arange(f_min_hz.value, f_max_hz.value, dnu.value) * u.Hz
frequency_band = np.linspace(f_min_hz.value, f_max_hz.value, 1000) * u.Hz
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
    'temp_ion': np.array([1.5e5]) * u.K, 
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([0]) * u.km/u.s,
}

enforce_units(quant)
# print("----->", compute_epsilon_ionized(quant.copy(), m_aqn_kg, frequency_band)["aqn_emit"] * (0.6*u.kpc).to(u.cm)/(4*np.pi))

# print("Initial temp_ion is::::")
# print(quant["temp_ion"])
# print("-------------------")

# Investigation of T_AQN VS dv, ioni_gas, m_aqn and T_gas_eff
# t_aqn_parameter_relations_study(quant.copy(), m_aqn_kg, frequency_band)

# Investigation of epsilon VS dv, ioni_gas, m_aqn and T_gas_eff
# epsilon_parameter_relations_study(quant.copy(), m_aqn_kg, frequency_band)


# Xunyu checks:
# print("Checking with Inu maps (22)")
# #============================================================================#
# print("---> Checking T_AQN")
# print("--> First estimate: 102.1 eV")
# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e4]) * u.K, 
#     'dv_ioni':  np.array([220]) * u.km/u.s, 
#     'dv_neut':  np.array([0]) * u.km/u.s,
# }
# enforce_units(quant)

# T_AQN = T_AQN_ionized3(
#     n_bar=quant["ioni_gas"], 
#     Dv=quant["dv_ioni"] * cst.c, 
#     f=1, 
#     g=0.1, 
#     T_p=quant["temp_ion"] * eV_to_K, 
#     R=calc_R_AQN(m_aqn_kg))
# print("-> I get, ", T_AQN)

# T_AQN = T_AQN_ionized2(
#     n_bar=quant["ioni_gas"], 
#     Dv=quant["dv_ioni"], 
#     f=1, 
#     g=0.1, 
#     T_p=quant["temp_ion"], 
#     R=calc_R_AQN(m_aqn_kg))
# print("-> I get, ", T_AQN)

# print("--> Second estimate: 4.623 eV")
# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1.5e5]) * u.K, 
#     'dv_ioni':  np.array([220]) * u.km/u.s, 
#     'dv_neut':  np.array([0]) * u.km/u.s,
# }
# enforce_units(quant)

# T_AQN = T_AQN_ionized3(
#     n_bar=quant["ioni_gas"], 
#     Dv=quant["dv_ioni"] * cst.c, 
#     f=1, 
#     g=0.1, 
#     T_p=quant["temp_ion"] * eV_to_K, 
#     R=calc_R_AQN(m_aqn_kg))
# print("-> I get, ", T_AQN)

# T_AQN = T_AQN_ionized2(
#     n_bar=quant["ioni_gas"], 
#     Dv=quant["dv_ioni"], 
#     f=1, 
#     g=0.1, 
#     T_p=quant["temp_ion"], 
#     R=calc_R_AQN(m_aqn_kg))
# print("-> I get, ", T_AQN)
# #============================================================================#
# print("---> Checking Phi")
# print("--> First estimate: 2.814e7 photon units")
# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e4]) * u.K, 
#     'dv_ioni':  np.array([220]) * u.km/u.s, 
#     'dv_neut':  np.array([0]) * u.km/u.s,
# }
# enforce_units(quant)

# from astropy import constants as cst
# from astropy import units as u
# lamb = 1500 * u.AA 
# nu = lamb.to(u.Hz, equivalencies=u.spectral()) # 1.999e15 [Hz]

# print("-> I get, ", compute_phi(quant.copy(), m_aqn_kg, nu))

# Epsilon = compute_epsilon_ionized(quant.copy(), m_aqn_kg, frequency_band, adjust_T_gas=False)["aqn_emit"]
# Phi = Epsilon * (0.6*u.kpc).to(u.cm)/(4*np.pi)
# print("-> I get, ", Phi)
# print("--> Second estimate: 257.3 photon units")
# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1.5e5]) * u.K, 
#     'dv_ioni':  np.array([220]) * u.km/u.s, 
#     'dv_neut':  np.array([0]) * u.km/u.s,
# }
# enforce_units(quant)

# from astropy import constants as cst
# from astropy import units as u
# lamb = 1500 * u.AA 
# nu = lamb.to(u.Hz, equivalencies=u.spectral()) # 1.999e15 [Hz]

# print("-> I get, ", compute_phi(quant.copy(), m_aqn_kg, nu))

# Epsilon = compute_epsilon_ionized(quant.copy(), m_aqn_kg, frequency_band, adjust_T_gas=False)["aqn_emit"]
# Phi = Epsilon * (0.6*u.kpc).to(u.cm)/(4*np.pi)
# print("-> I get, ", Phi)
# #============================================================================#

print("---> Checking Phi (integrated), i.e. <Phi>, which is Inu maps (24)")

print(nu_range.to(u.AA, equivalencies=u.spectral()))
# print(frequency_band[0].to(u.AA, equivalencies=u.spectral()))
# print(frequency_band[-1].to(u.AA, equivalencies=u.spectral()))
# w = nu.to(u.AA, equivalencies=u.spectral())

quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4]) * u.K, 
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([0]) * u.km/u.s,
}
enforce_units(quant)

T_AQN = T_AQN_ionized3(
    n_bar=quant["ioni_gas"], 
    Dv=quant["dv_ioni"] * cst.c, 
    f=1, 
    g=0.1, 
    T_p=quant["temp_ion"] * eV_to_K, 
    R=calc_R_AQN(m_aqn_kg))


lambda_0 = 1500 * u.AA
dlambda = 400 * u.AA
x0 = (2*np.pi*cst.hbar*cst.c)/(T_AQN*lambda_0)
sinh_arg = (x0*dlambda/(2*lambda_0)).to(u.dimensionless_unscaled).value

ratio_theory = (1+0.005926)*(2*lambda_0/(x0*dlambda))*np.sinh(sinh_arg)
ratio_theory = ratio_theory.to(u.dimensionless_unscaled)
print(x0.to(u.dimensionless_unscaled))
print("--> Ratio estimate is:", ratio_theory.value)


from astropy import constants as cst
from astropy import units as u
lamb = 1500 * u.AA 
nu = lamb.to(u.Hz, equivalencies=u.spectral()) # 1.999e15 [Hz]
Phi_no_band = compute_phi(quant.copy(), m_aqn_kg, nu)

Epsilon = compute_epsilon_ionized_bandwidth_2(quant.copy(), m_aqn_kg, frequency_band, adjust_T_gas=False)["aqn_emit"]
Phi_band = Epsilon * (0.6*u.kpc).to(u.cm)/(4*np.pi)

ratio = Phi_band / Phi_no_band

print("-> Calculated ratio is:", ratio.value)
print("-> Difference is: ", (ratio.value/ratio_theory.value-1)*100,"%")
#============================================================================#
'''
print("--> Now plotting checks for <Phi> / Phi")

T_gas_array = np.logspace(2, 7, 100) * u.K
T_AQN_array = np.zeros(len(T_gas_array)) * u.eV
ratio_theory_array = np.zeros(len(T_gas_array))
ratio_array = np.zeros(len(T_gas_array))

for i in range(len(T_gas_array)):

    quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': T_gas_array[i],
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([0]) * u.km/u.s,
    }
    enforce_units(quant)

    T_AQN = T_AQN_ionized3(
        n_bar=quant["ioni_gas"], 
        Dv=quant["dv_ioni"] * cst.c, 
        f=1, 
        g=0.1, 
        T_p=quant["temp_ion"] * eV_to_K, 
        R=calc_R_AQN(m_aqn_kg))

    T_AQN_array[i] = T_AQN[0]

    lambda_0 = 1500 * u.AA
    dlambda = 400 * u.AA
    x0 = (2*np.pi*cst.hbar*cst.c)/(T_AQN*lambda_0)
    sinh_arg = (x0*dlambda/(2*lambda_0)).to(u.dimensionless_unscaled).value

    ratio_theory = (1+0.005926)*(2*lambda_0/(x0*dlambda))*np.sinh(sinh_arg)
    ratio_theory = ratio_theory.to(u.dimensionless_unscaled)

    ratio_theory_array[i] = ratio_theory[0].value
    # print("--> Ratio estimate is:", ratio_theory.value)


    from astropy import constants as cst
    from astropy import units as u
    lamb = 1500 * u.AA 
    nu = lamb.to(u.Hz, equivalencies=u.spectral()) # 1.999e15 [Hz]
    Phi_no_band = compute_phi(quant.copy(), m_aqn_kg, nu)
    # print(T_gas_array[i], Phi_no_band.value, end="\t")
    Epsilon = compute_epsilon_ionized_bandwidth_2(quant.copy(), m_aqn_kg, frequency_band, adjust_T_gas=False)["aqn_emit"]
    Phi_band = Epsilon * (0.6*u.kpc).to(u.cm)/(4*np.pi)
    # print(Phi_band.value)
    ratio = Phi_band / Phi_no_band
    ratio_array[i] = ratio[0].value

    # print("-> Calculated ratio is:", ratio.value)

# print(ratio_array)
plt.figure(dpi=100)
plt.plot(T_AQN_array, ratio_array, label="Computed ratio", linewidth=3)
plt.plot(T_AQN_array, ratio_theory_array, label="Theory using Inu maps (24)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T_AQN [eV]")
plt.ylabel("<Phi>/Phi")
plt.legend()
plt.show()

x0_array = ((2*np.pi*cst.hbar*cst.c)/(T_AQN_array*lambda_0)).to(u.dimensionless_unscaled).value

plt.figure(dpi=100)
plt.plot(x0_array, ratio_array, label="Computed ratio", linewidth=3)
plt.plot(x0_array, ratio_theory_array, label="Theory using Inu maps (24)")
plt.axvline(x=1,color="red", label="x0=1")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("x0")
plt.ylabel("<Phi>/Phi")
plt.legend()
plt.show()

for i in range(len(x0_array)):
    print(x0_array[i], "\t", ratio_theory_array[i], "\t", ratio_array[i],
        "\t", ratio_theory_array[i]/ratio_array[i], 
        "\t", ratio_theory_array[i]-ratio_array[i])
'''
#============================================================================#

print("---> Checking Phi (integrated) with velocity distribution, which is Inu maps (30)")

# plot_maxwell_boltzmann()

quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4]) * u.K, 
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([0]) * u.km/u.s,
}
enforce_units(quant)
from astropy import constants as cst
from astropy import units as u
lamb = 1500 * u.AA 
nu = lamb.to(u.Hz, equivalencies=u.spectral()) # 1.999e15 [Hz]
epsilon = compute_epsilon_ionized_bandwidth(quant.copy(), m_aqn_kg, frequency_band)["aqn_emit"]
Phi_no_distribution = epsilon * (0.6*u.kpc).to(u.cm)/(4*np.pi)
# print(Phi_no_distribution)


def compute_epsilon_integrand(v, quant, sigma_v, v_b, m_aqn_kg, frequency_band):
    f_res = f_maxbolt(v, sigma_v, v_b)
    quant_copy = quant.copy()
    quant_copy["dv_ioni"] = (v*u.km/u.s) / cst.c.to(u.km/u.s)

    e_res = compute_epsilon_ionized(quant_copy, m_aqn_kg, frequency_band)["aqn_emit"].value # _bandwidth

    return e_res * f_res

# print(compute_epsilon_integrand(0.5, quant, 156, 180, m_aqn_kg, frequency_band))


def compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, sigma_v, v_b):
    # result, error = quad(f_maxbolt, 0, 10000, args=(sigma_v.value, v_b.value))
    result, error = quad(compute_epsilon_integrand, 0.1, 800, 
        args=(quant, sigma_v.value, v_b.value, m_aqn_kg, frequency_band))

    # print(f"Integral result: {result:.4e}, Estimated error: {error:.4e}")

    return result * epsilon_units

def compute_epsilon_integrand_bandwidth(v, quant, sigma_v, v_b, m_aqn_kg, frequency_band):
    f_res = f_maxbolt(v, sigma_v, v_b)
    quant_copy = quant.copy()
    quant_copy["dv_ioni"] = (v*u.km/u.s) / cst.c.to(u.km/u.s)

    e_res = compute_epsilon_ionized_bandwidth_2(quant_copy, m_aqn_kg, frequency_band)["aqn_emit"].value # _bandwidth

    return e_res * f_res

# print(compute_epsilon_integrand(0.5, quant, 156, 180, m_aqn_kg, frequency_band))


def compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, sigma_v, v_b):
    # result, error = quad(f_maxbolt, 0, 10000, args=(sigma_v.value, v_b.value))
    result, error = quad(compute_epsilon_integrand_bandwidth, 0, 5*(sigma_v.value+v_b.value), 
        args=(quant, sigma_v.value, v_b.value, m_aqn_kg, frequency_band))

    # print(f"Integral result: {result:.4e}, Estimated error: {error:.4e}")

    return result * epsilon_units

sigma_v = 110 * u.km/u.s
v_b = 120 * u.km/u.s
epsilon_res = compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, sigma_v, v_b)
phi = epsilon_res * (0.6*u.kpc).to(u.cm)/(4*np.pi)
print(phi)


# sigma_v_array = np.linspace(100,800, 100) * u.km/u.s
# phi_array = np.zeros(len(sigma_v_array)) * photon_units * u.sr
# for i in range(len(sigma_v_array)):
#     epsilon_res = compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, sigma_v_array[i], v_b)
#     phi_array[i] = epsilon_res * (0.6*u.kpc).to(u.cm)/(4*np.pi)

# plt.figure(dpi=300)
# plt.plot(sigma_v_array, phi_array, label="asdf", linewidth=3)
# # plt.plot(sigma_v_array, sigma_v_array**3/sigma_v_array[0] * phi_array[0])
# # plt.plot(sigma_v_array, ratio_theory_array, label="Theory using Inu maps (24)")
# # plt.axvline(x=1,color="red", label="x0=1")
# # plt.xscale("log")
# # plt.yscale("log")
# plt.xlabel("sigma_v [km/s]")
# plt.ylabel("Phi (no bandwidth integral)")
# # plt.legend()
# plt.show()


quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4]) * u.K, 
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([0]) * u.km/u.s,
}
enforce_units(quant)

print("------------------------------------------------------")

print("sigma_v = 50, v_b = 50, Phi = 40.22")
print(" --> ", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 
    50*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 100, v_b = 50, Phi = 7.524")
print(" --> ", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 
    100*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 110, v_b = 180, Phi = 1.702")
print(" --> ", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 
    110*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 156, v_b = 180, Phi = 1.167")
print(" --> ", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 
    156*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 50, v_b = 180, Phi = 0.1312")
print(" --> ", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 
    50*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 500, v_b = 180, Phi = 0.06485")
print(" --> ", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 
    500*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 110, v_b = 500, Phi = 2.29831e-4")
print(" --> ", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 
    110*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 500, v_b = 500, Phi = 0.04198")
print(" --> ", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 
    500*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))

print("------------------------------------------------------")



'''
Checking parameters v.s. FIRE simulation
Using lit values of velocity -> don't use velocity from FIRE
> 1 kg ruled out from Fereshteh paper
Produce Phi per wavelength plot to check w/ Jayant's ARCADE excess
'''




# print("sigma_v = 110, v_b = 180, Phi = 1.702, Phi_(30) = 229.5")
# print("Without bandwith integral -->", compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 110*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("With    bandwith integral -->", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 110*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("\n")

# print("sigma_v = 156, v_b = 180, Phi = 1.167, Phi_(30) = 229.5")
# print("Without bandwith integral -->", compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 156*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("With    bandwith integral -->", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 156*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("\n")

# print("sigma_v = 110, v_b = 120, Phi = 3.556, Phi_(30) = 229.5")
# print("Without bandwith integral -->", compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 110*u.km/u.s, 120*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("With    bandwith integral -->", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 110*u.km/u.s, 120*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("\n")

# print("sigma_v = 110, v_b = 220, Phi = 0.8837, Phi_(30) = 229.5")
# print("Without bandwith integral -->", compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 110*u.km/u.s, 220*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("With    bandwith integral -->", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 110*u.km/u.s, 220*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("\n")

# print("sigma_v = 110, v_b = 500, Phi = 2.298e-4, Phi_(30) = 229.5")
# print("Without bandwith integral -->", compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 110*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("With    bandwith integral -->", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 110*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("\n")

# print("sigma_v = 11, v_b = 180, Phi = 4.266e-14, Phi_(30) = 2.295e5 ")
# # print("-->", compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 11*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("Integration failed, overflow error. Need to investigate.")
# print("\n")

# print("sigma_v = 11, v_b = 18, Phi = 1246, Phi_(30) = 2.295e5")
# print("Without bandwith integral -->", compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 11*u.km/u.s, 18*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("With    bandwith integral -->", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 11*u.km/u.s, 18*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("\n")

# print("sigma_v = 500, v_b = 500, Phi = 4.198e-2, Phi_(30) = 2.444")
# print("Without bandwith integral -->", compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 500*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# print("With    bandwith integral -->", compute_epsilon_integrated_bandwidth(quant, m_aqn_kg, frequency_band, 500*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


# 5 (sigma_v + v_b)


# print(compute_epsilon_integrated(quant, m_aqn_kg, frequency_band, 156*u.km/u.s, 180*u.km/u.s))
# print(compute_epsilon_ionized(quant, m_aqn_kg, frequency_band)["aqn_emit"][0])

# def parameter_variation_integrand(quant, sigma_v, v_b, m_aqn_kg, frequency_band, parameter_name, parameter_array, mass_variation = False):
#     parameter_array_length = len(parameter_array)

#     epsilon_array = np.zeros(parameter_array_length)

#     if not mass_variation:
#         for i, parameter in enumerate(parameter_array):
#             quant[parameter_name] = parameter
#             epsilon_array[i] = compute_epsilon_integrand(quant["dv_ioni"]*cst.c.to(u.km/u.s).value, quant, sigma_v, v_b, m_aqn_kg, frequency_band)[0]
#     else:
#         for i, parameter in enumerate(parameter_array):
#             epsilon_array[i] = compute_epsilon_integrand(quant["dv_ioni"]*cst.c.to(u.km/u.s).value, quant, sigma_v, v_b, parameter, frequency_band)[0]

#     return epsilon_array * epsilon_units


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
