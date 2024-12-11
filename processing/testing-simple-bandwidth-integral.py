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

#======================================================================#
def h(x):
    return np.where(
        x < 0,
        0,
        np.where(x < 1, 17 - 12 * np.log(x / 2), 17 + 12 * np.log(2)))

def H(x):
    return (1+x)*np.exp(-x)*h(x)
#======================================================================#
def func(lamb, x0):
	kT = ((2*np.pi*cst.hbar*cst.c)/(x0*lambda0)).to(u.J)
	x = ((2*np.pi*cst.hbar*cst.c)/(kT*lamb*u.AA)).to(u.dimensionless_unscaled)
	return H(x)/H(x0)*1/lamb
def func(lamb, x0):
	kT = ((2*np.pi*cst.hbar*cst.c)/(x0*lambda0)).to(u.J)
	x = ((2*np.pi*cst.hbar*cst.c)/(kT*lamb*u.AA)).to(u.dimensionless_unscaled)
	return lambda0.value*H(x)/H(x0)*1/lamb
#======================================================================#
def integrate_func(func, band, x0, option="sum"):
	lamb_range = np.max(band) - np.min(band)

    #--> different integration options:

    #--> regular sum
	if option=="sum":
	    dlamb = band[1] - band[0]
	    integral = np.sum(func(band, x0) * dlamb)

    #--> scipy.integrate.quad
	if option=="quad":
		integral = quad(func, np.min(band), np.max(band), args=(x0),
			epsabs=1e-9, epsrel=1e-9)[0]

	#--> scipy.special.expi
	if option=="nquad":
		integral = nquad(func, [[np.min(band), np.max(band)]], args=(x0))

	return 1 / lamb_range * integral
#======================================================================#
# def integrate_func(func, band, x0):
#       dlamb = band[1] - band[0]
#       lamb_range = np.max(band) - np.min(band)
#       integral = np.sum(func(band, x0) * dlamb / band)
#       return lambda0.value / lamb_range * integral

# def bandwidth_integrate_func(func, band, x0):
#       dlamb = band[1] - band[0]
#       lamb_range = np.max(band) - np.min(band)
#       integral = np.sum(func(band, x0) * dlamb)
#       return 1 / lamb_range * integral
#======================================================================#
dband = 10000000
band = np.linspace(1300,1700,dband)
lambda0 = 1500 * u.AA

x0_array = [0.0005, 0.001, 0.01, 0.1, 0.5, 0.75, 1, 10, 100]	
ratio_theory_array = [1.00506, 1.00499, 1.00464, 1.00382, 1.00087, 0.998927, 1.01443, 
1.14001, 4829.44]

def study(option):
	print("------------------------ "+option+" ------------------------")
	print("       x0     |     Xunyu     |    Integral   |     Diff     |")
	for i in range(len(x0_array)):
		
		plt.plot(band, func(band, x0_array[i]))

		integral = integrate_func(func, band, x0_array[i], option)

		percent_off = (integral-ratio_theory_array[i])*100

		print(f"{x0_array[i]:<15.6f} {ratio_theory_array[i]:<15.6f} {integral:<15.6f} {percent_off:<12.6f}%")
	plt.yscale("log")
	plt.show()
#======================================================================#

# study("sum")
# study("quad")
'''
x_plot = np.linspace(0.001, 150)
x0_array = np.logspace(-4, 2, 10)
for i in range(len(x0_array)):

	kT = ((2*np.pi*cst.hbar*cst.c)/(x0_array[i]*lambda0)).to(u.J)
	x = ((2*np.pi*cst.hbar*cst.c)/(kT*band*u.AA)).to(u.dimensionless_unscaled)

	plt.plot(x_plot, H(x_plot)/H(x0_array[i]), "--", color="black", alpha=0.5, linewidth=0.5)
	plt.plot(x, H(x)/H(x0_array[i]), label=f"x0={x0_array[i]:.4f}")
plt.yscale("log")
plt.legend(loc='upper right')
plt.xlabel("x")
plt.ylabel("H(x)/H(x0)")
plt.ylim(1e-7, 1e7)
plt.show()
'''



def compute_epsilon_ionized_bandwidth_2(cubes_import, m_aqn_kg, frequency_band, adjust_T_gas=True):

    #------------------------------------------------------#
    cubes = cubes_import.copy()
    enforce_units(cubes)
    R_aqn_cm = calc_R_AQN(m_aqn_kg)
    cubes["dark_mat"] = cubes["dark_mat"] * 2/5
    #------------------------------------------------------#

    #------------------------------------------------------#
    # compute effective gas temperature
    if adjust_T_gas:
        cubes["temp_ion_eff"] = cubes["temp_ion"] + 1/2 * cst.m_p * kg_to_eV * cubes["dv_ioni"]**2
    else:
        cubes["temp_ion_eff"] = cubes["temp_ion"] 

    # compute AQN temperature
    cubes["t_aqn_i"] = T_AQN_ionized2(  cubes["ioni_gas"], cubes["dv_ioni"], f, g, 
                                        cubes["temp_ion_eff"], R_aqn_cm)
    #------------------------------------------------------#

    def spectral_surface_emissivity_no_H(T_in):
        T = T_in * eV_to_erg
        unit_factor = (1 / cst.hbar.cgs) * (1/(cst.hbar.cgs * cst.c.cgs))**2 * (cst.hbar.cgs * 1/u.Hz * 1/u.s)
        #                ^ 1/seconds           ^ 1/area                          ^ 1/frequency and energy  

        return unit_factor * 4/45 * T**3 * cst.alpha ** (5/2) * 1/np.pi * (T/(m_e_eV*eV_to_erg))**(1/4)

    def integrate_func(func, band, x0):
        lamb_range = np.max(band) - np.min(band)

        integral = quad_vec(func, np.min(band), np.max(band), args=(x0.value,),
            epsabs=1e-9, epsrel=1e-9)[0]

        return 1 / lamb_range * integral * photon_units/erg_hz_cm2/u.sr

    def func(lamb, x0):
        kT = ((2*np.pi*cst.hbar*cst.c)/(x0*lambda0)).to(u.J)
        x = ((2*np.pi*cst.hbar*cst.c)/(kT*lamb*u.AA)).to(u.dimensionless_unscaled)

        C = (erg_hz_cm2).to(photon_units*u.sr, u.spectral_density(lamb*u.AA))
        to_skymap_units_conversion = C / erg_hz_cm2 * 2*np.pi

        return lambda0.value*H(x)*1/lamb * to_skymap_units_conversion.value/(dOmega.value)

    dband = 1000000
    band = np.linspace(1300,1700,dband)
    lambda0 = 1500 * u.AA
    x0 = ((2*np.pi*cst.hbar*cst.c)/(cubes["t_aqn_i"]*lambda0)).to(u.dimensionless_unscaled)

    cubes["aqn_emit"] = spectral_surface_emissivity_no_H(cubes["t_aqn_i"]) * integrate_func(func, band, x0) 

    cubes["aqn_emit"] = cubes["aqn_emit"] * 4 * np.pi * R_aqn_cm**2 * \
                       (cubes["dark_mat"] / m_aqn_kg).to(1/u.cm**3) * u.sr

    return cubes



# print(compute_epsilon_integrand(0.5, quant, 156, 180, m_aqn_kg, frequency_band))


def compute_epsilon_velocity_integral(quant, m_aqn_kg, frequency_band, adjust_T_gas, sigma_v, v_b):

	def epsilon_velocity_integrand(v, quant, sigma_v, v_b, m_aqn_kg, frequency_band, adjust_T_gas):
	    f_res = f_maxbolt(v, sigma_v, v_b)
	    quant_copy = quant.copy()
	    quant_copy["dv_ioni"] = (v*u.km/u.s) / cst.c.to(u.km/u.s)

	    e_res = compute_epsilon_ionized_bandwidth_2(quant_copy, m_aqn_kg, frequency_band, adjust_T_gas)["aqn_emit"].value # _bandwidth

	    return e_res * f_res

	result, error = quad_vec(epsilon_velocity_integrand, 0, 5*(sigma_v.value+v_b.value), 
        args=(quant, sigma_v.value, v_b.value, m_aqn_kg, frequency_band, adjust_T_gas))

	return result * epsilon_units

sigma_v = 110 * u.km/u.s
v_b = 120 * u.km/u.s

m_aqn_kg = 1e-2*u.kg
frequency_band = np.linspace(f_min_hz.value, f_max_hz.value, 1000) * u.Hz



quant = {
    'dark_mat': np.array([0.3,0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01,0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0,0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e6,1e4]) * u.K, 
    'dv_ioni':  np.array([220,100]) * u.km/u.s, 
    'dv_neut':  np.array([0,0]) * u.km/u.s,
}

# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e6]) * u.K, 
#     'dv_ioni':  np.array([120]) * u.km/u.s, 
#     'dv_neut':  np.array([0]) * u.km/u.s,
# }

enforce_units(quant)


res = compute_epsilon_ionized_bandwidth_2(quant, m_aqn_kg, frequency_band, adjust_T_gas=False)
# print(res)
print(res["aqn_emit"]*(0.6*u.kpc).to(u.cm)/(4*np.pi))


res = compute_epsilon_velocity_integral(quant, m_aqn_kg, frequency_band, False, sigma_v, v_b)
print(res*(0.6*u.kpc).to(u.cm)/(4*np.pi))

# x0 = np.array([10,1])
# kT = ((2*np.pi*cst.hbar*cst.c)/(x0*lambda0)).to(u.J)
# lamb = 100
# x = ((2*np.pi*cst.hbar*cst.c)/(kT*lamb*u.AA)).to(u.dimensionless_unscaled)

# print(H())








# # WORK WITH A SINGLE INTEGRAL:
# x0 = 0.0005
# lambda0 = 1500 * u.AA

# x_min = ((2*np.pi*cst.hbar*cst.c)/(kT*1800*u.AA)).to(u.dimensionless_unscaled)
# x_max = ((2*np.pi*cst.hbar*cst.c)/(kT*1300*u.AA)).to(u.dimensionless_unscaled)



# true_value = 1.00506
# print(integral, true_value, (integral/true_value-1)*100,"%")

# PRINT A BUNCH OF VALUES FOR XUNYU










