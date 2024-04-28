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








def h(x):
    if x < 1:
        return (17 - 12*np.log(x/2))
    else:
        return (17 + 12*np.log(2))


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
def spectral_surface_emissivity(nu, T):
    T = T * eV_to_erg
    w = 2 * np.pi * nu * Hz_to_erg
    unit_factor = (1 / cst.hbar.cgs) * (1/(cst.hbar.cgs * cst.c.cgs))**2 * (cst.hbar.cgs * 1/u.Hz * 1/u.s)
    #                ^ 1/seconds           ^ 1/area                          ^ 1/frequency and energy
    return unit_factor * 4/45 * T**3 * cst.alpha ** (5/2) * 1/np.pi * (T/(m_e_eV*eV_to_erg))**(1/4) * (1 + w/T) * np.exp(- w/T) * h(w/T)



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
