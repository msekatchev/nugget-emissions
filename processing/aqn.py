import numpy as np
from astropy import constants as cst
from astropy import units as u
from constants import *
from scipy.optimize import fsolve
import collections.abc
print("Loaded AQN script")


# cm
# R cm
# T_AQN eV
# T_gas eV
def R_eff(R,T_AQN,T_gas):
    return np.sqrt(8 * (m_e_eV * eV_to_erg) * (R * cm_to_inverg)**4 * (T_AQN*eV_to_erg)**3 / (np.pi * (T_gas*eV_to_erg)**2)) * 1/cm_to_inverg




# eV
# n_bar m^-3 or cm^-3
# Dv unitless
# f unitless
# g unitless
def T_AQN_analytical(n_bar, Dv, f, g):
    return ((1-g) * np.pi * 3/16 * 1/(cst.alpha**(5/2)) * (m_e_erg)**(1/4) * 
            (E_ann_GeV * GeV_to_erg) * f * Dv * ((n_bar).to(1/u.cm**3) * invcm_to_erg**3))**(4/17) * erg_to_eV

# eV
# n_bar m^-3 or cm^-3
# Dv unitless
# f unitless
# g unitless
# T_p eV
# R cm
def T_AQN_ionized(n_bar, Dv, f, g, T_p, R):
    return (3/8 * E_ann_GeV * GeV_to_erg  * (1-g) * f * Dv * n_bar.to(1/u.cm**3) * 1/cm_to_inverg**3 * R**2 * cm_to_inverg**2 * (1/cst.alpha)**(5/2) * 1/T_p**2 * 1/eV_to_erg**2)**(4/5)* (m_e_erg) * erg_to_eV


def T_AQN_ionized2(n_bar, Dv, f, g, T_p, R):
    f=1
    n_bar_erg = n_bar.to(1/u.cm**3) * 1/cm_to_inverg**3
    return (3/4 * np.pi * (1-g) * f * Dv * 1/cst.alpha**(3/2) * m_p_erg * n_bar_erg * 
            R**2 * cm_to_inverg**2 * 
            1/T_p**2 * 1/eV_to_erg**2)**(4/7)* (m_e_erg) * erg_to_eV


# erg s^-1 cm^-2
# rate of annihilation F_ann
# n_bar m^-3 or cm^-3 or GeV^3
# Dv unitless
# f unitless
def F_ann(n_bar, Dv, f):                   # dE [erg] / dt [s] dA [cm^2]
    unit_factor = 1
    if(E_ann_GeV.unit == "GeV"):
        unit_factor *= GeV_to_erg
    if(n_bar.unit == "GeV^3"):
        unit_factor *= 1/GeVinv_to_cm**3
    if(Dv.unit == ""):
        unit_factor *= cst.c.cgs
    return unit_factor * E_ann_GeV * f * Dv * n_bar.to(1/u.cm**3)


# erg s^-1 cm^-2
# T eV
def F_tot(T):
    unit_factor = eV_to_erg**4 / inverg_to_cm**2 * 1/cst.hbar.cgs
    return unit_factor * 16/3 * T**4 * cst.alpha**(5/2) * 1/np.pi * (T/m_e_eV)**(1/4) 
                                     # alpha = fine-structure constant

# eV
# x eV
# data -> n_bar m^-3 or cm^-3 or GeV^3
#         Dv unitless
#         f unitless
#         g unitless
def T_numerical_func(x, *data):
    n_bar = data[0]
    a = F_tot(x*u.eV) - (1-g)*F_ann(n_bar,Dv,f)*np.ones(len(x)) # Correction: the (1-g) factor should be on F_ann side as in the paper
    return a

# eV
# data -> n_bar m^-3 or cm^-3 or GeV^3
#         Dv unitless
#         f unitless
#         g unitless
def T_AQN_numerical(n_bar, Dv, f, g):
    return fsolve(T_numerical_func, 1,args=(n_bar,Dv,f,g))[0]*u.eV



# print("T_AQN is: ", T_AQN_analytical(n_bar, Dv, f, g))







# simple h function without array operations
# def h(x):
#     if x < 1:
#         return (17 - 12*np.log(x/2))
#     else:
#         return (17 + 12*np.log(2))

# h function with array operations
h_func_cutoff = 17 + 12*np.log(2)
# def h(x):
#     return_array = np.copy(x)
#     return_array[np.where(x<1)] = (17 - 12*np.log(x[np.where(x<1)]/2))
#     return_array[np.where(x>=1)] = h_func_cutoff
#     return return_array

# updated function below accounts for 0 X values
def h(x):
    return_array = np.copy(x)
    try:
        return_array[np.where(x<=0)] = 0
        return_array[np.where((x<1) & (x>0))] = (17 - 12*np.log(x[np.where(x<1)]/2))
        return_array[np.where(x>=1)] = h_func_cutoff
    except:
        if x < 0:
            return 0
        else:
            if x < 1:
                return (17 - 12*np.log(x/2))
            else:
                return (17 + 12*np.log(2))        
    return return_array

def H(x):
    return (1+x)*np.exp(-x)*h(x)


def h(x):
    return x.copy()


# def h(x):
#     if type(x) == np.ndarray or len(np.array([x]))>1: #isinstance(x, (np.ndarray)) and (x*u.eV).unit == u.eV:
#         return_array = np.zeros(len(x))
#         return_array[np.where(x<1)] = (17 - 12*np.log(x[np.where(x<1)]/2))
#         return_array[np.where(x>=1)] = 17 + 12*np.log(2)
#         return return_array
#     else:
#         if x < 1:
#             return (17 - 12*np.log(x/2))
#         else:
#             return (17 + 12*np.log(2))

m_e_eV  = (cst.m_e.cgs*cst.c.cgs**2).value * u.erg * erg_to_eV  # mass of electron    in eV

# erg Hz^-1 s^-1 cm^-2
# nu Hz
# T eV
# def spectral_surface_emissivity(nu, T):
#     T = T * eV_to_erg
#     w = 2 * np.pi * nu * Hz_to_erg
#     unit_factor = (1 / cst.hbar.cgs) * (1/(cst.hbar.cgs * cst.c.cgs))**2 * (cst.hbar.cgs * 1/u.Hz * 1/u.s)
#     #                ^ 1/seconds           ^ 1/area                          ^ 1/frequency and energy
#     return unit_factor * 4/45 * T**3 * cst.alpha ** (5/2) * 1/np.pi * (T/(m_e_eV*eV_to_erg))**(1/4) * (1 + w/T) * np.exp(- w/T) * h(w/T)

# updated function below accounts for 0 T values:
def spectral_surface_emissivity(nu_in, T_in):
    nu, T = nu_in.copy(), T_in.copy()
    T = T * eV_to_erg
    w = 2 * np.pi * nu * Hz_to_erg
    unit_factor = (1 / cst.hbar.cgs) * (1/(cst.hbar.cgs * cst.c.cgs))**2 * (cst.hbar.cgs * 1/u.Hz * 1/u.s)
    #                ^ 1/seconds           ^ 1/area                          ^ 1/frequency and energy
    X = w/T
    X[T<=0] = 0

    return unit_factor * 4/45 * T**3 * cst.alpha ** (5/2) * 1/np.pi * (T/(m_e_eV*eV_to_erg))**(1/4) * (1 + X) * np.exp(- X) * h(X)


# erg Hz^-1 s^-1 cm^-3
# n_AQN m^-3
# n_bar m^-3
# Dv unitless
# f unitless
# g unitless
# nu Hz
def spectral_spatial_emissivity(n_AQN, n_bar, Dv, f, g, nu):
    T_AQN = T_AQN_analytical(n_bar, Dv, f, g)
    #T_AQN = 1 * u.eV
    dFdw = spectral_surface_emissivity(nu, T_AQN)
    return dFdw * 4 * np.pi * R_AQN**2 * n_AQN.to(1/u.cm**3)


def compute_epsilon_ionized(cubes_import, m_aqn_kg, frequency_band):
    dnu = frequency_band[1] - frequency_band[0]
    nu_range = np.max(frequency_band) - np.min(frequency_band)

    cubes = cubes_import.copy()
    enforce_units(cubes)
    # dark_mat, ioni_gas, neut_gas, temp_ion, dv_ioni, dv_neut = \
    # cubes["dark_mat"], cubes["ioni_gas"], cubes["neut_gas"], cubes["temp_ion"], cubes["dv_ioni"], cubes["dv_neut"]

    R_aqn_cm = calc_R_AQN(m_aqn_kg)

    cubes["dark_mat"] = cubes["dark_mat"] * 2/5

    # compute effective gas temperature
    cubes["temp_ion_eff"] = cubes["temp_ion"] # + 1/2 * cst.m_p * kg_to_eV * cubes["dv_ioni"]**2

    # print(cubes["temp_ion_eff"])

    # compute AQN temperature
    cubes["t_aqn_i"] = T_AQN_ionized2(  cubes["ioni_gas"], cubes["dv_ioni"], f, g, 
                                        cubes["temp_ion_eff"], R_aqn_cm)
    
    # print(cubes["temp_ion"])
    # print(cubes["temp_ion_eff"])

    # cubes["t_aqn_i"] = np.ones(cubes["t_aqn_i"].shape) * u.eV

    # from erg/s/Hz/cm2 to photons/s/A/cm2
    def to_skymap_units(F_erg_hz_cm2,nu):

        w = nu.to(u.AA, equivalencies=u.spectral())
        C = (erg_hz_cm2).to(photon_units*u.sr, u.spectral_density(w))

        return F_erg_hz_cm2 * C / erg_hz_cm2 * 2*np.pi

    cubes["aqn_emit"] = np.zeros(np.shape(cubes["t_aqn_i"])) * photon_units
    
    for nu in frequency_band:
        cubes["aqn_emit"] += to_skymap_units(spectral_surface_emissivity(nu, 
                           cubes["t_aqn_i"])/(dOmega)*dnu/nu_range, nu)        

    # vvv omit frequency band integration, use mean of frequency band instead: vvv
    # nu_mean = np.mean(frequency_band)
    # cubes["aqn_emit"] = to_skymap_units(spectral_surface_emissivity(nu_mean, 
    #                        cubes["t_aqn_i"])/(dOmega), nu_mean) 

    cubes["aqn_emit"] = cubes["aqn_emit"] * 4 * np.pi * R_aqn_cm**2 * \
                       (cubes["dark_mat"] / m_aqn_kg).to(1/u.cm**3) * u.sr

    return cubes

# Define the function f(v) based on the Maxwell-Boltzmann distribution with a velocity shift v_b
def f_maxbolt(v, sigma_v=156, v_b=180):
    # v, sigma_v, v_b = v, sigma_v.value, v_b.value
    prefactor = 4 * np.pi * v**2 * (1 / (2 * np.pi * sigma_v**2))**(3/2)
    exponential = np.exp(-(v**2 + v_b**2) / (2 * sigma_v**2))
    
    if v_b == 0:
        return prefactor * np.exp(-v**2 / (2 * sigma_v**2))
    
    sinh_term = np.sinh(v * v_b / sigma_v**2) / (v * v_b / sigma_v**2)
    #with np.errstate(divide='ignore', invalid='ignore'):
    #    sinh_term = np.sinh(v * v_b / sigma_v**2) / (v * v_b / sigma_v**2)
    #    sinh_term[v == 0] = 1  # Handle division by zero case
    
    return prefactor * exponential * sinh_term


