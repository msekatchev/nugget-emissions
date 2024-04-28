import numpy as np
from astropy import constants as cst
from astropy import units as u
print("Loaded constants script")
# Constants

# ----------------------------- unit conversions ----------------------------- 
eV_to_erg = 1e7*cst.e.si.value*u.erg/u.eV
GeV_to_erg = eV_to_erg.to(u.erg/u.GeV)
erg_to_eV = 1/eV_to_erg
erg_to_MeV = erg_to_eV.to(u.MeV/u.erg)
erg_to_GeV = erg_to_eV.to(u.GeV/u.erg)

cm_to_inverg = 1/(cst.hbar.cgs*cst.c.cgs)
inverg_to_cm = 1/cm_to_inverg
m_to_inverg = 1/(cst.hbar.cgs*cst.c.cgs) * 100 * u.cm / u.m                                                                                                             #              !!!
invm3_to_GeV3 = 1 / m_to_inverg**3 * erg_to_GeV**3                                                                                                                      #              !!!

kpc_to_cm = (u.kpc).to(u.cm) * u.cm / u.kpc

#g_to_eV = cst.c.cgs**2*erg_to_eV # this is wrong
K_to_eV = cst.k_B.cgs*erg_to_eV
eV_to_K = 1/K_to_eV
Hz_to_eV = cst.hbar.cgs*erg_to_eV
Hz_to_erg = cst.hbar.cgs * 1/u.s * 1/u.Hz #                                                                                is this correct?
cm_to_GeVinv = 1e7*cst.e.si.value*1e9/(cst.hbar.cgs.value*cst.c.cgs.value)/u.GeV/u.cm
cm_to_eVinv = 1e7*cst.e.si.value/(cst.hbar.cgs.value*cst.c.cgs.value)/u.eV/u.cm
GeVinv_to_cm = 1/cm_to_GeVinv
eVinv_to_cm = 1/cm_to_eVinv
1/u.GeV*GeVinv_to_cm  #  1 GeV^(-1) in cm
g_to_GeV = cst.c.cgs.value**2/(1e7*cst.e.si.value*1e9)*u.GeV/u.g
GeV_to_g = 1/g_to_GeV

invcm_to_erg = 1/cm_to_GeVinv * GeV_to_erg



# ----------------------------- functions -----------------------------

def sech(x):
    return 1 / np.cosh(x)

#nuclear_density_cgs = (2.3e17 * u.kg/u.m**3).cgs
nuclear_density_cgs = (3.5e17 * u.kg/u.m**3).cgs

def calc_m_AQN(R):
    return 4/3 * np.pi * R.cgs**3 * nuclear_density_cgs
def calc_R_AQN(m):
    return (3/4 * m.cgs/nuclear_density_cgs * 1/np.pi)**(1/3)

# ----------------------------- constants ----------------------------- 


m_e_eV  = (cst.m_e.cgs*cst.c.cgs**2).value * u.erg * erg_to_eV  # mass of electron    in eV
m_e_erg = (cst.m_e.cgs*cst.c.cgs**2).value * u.erg              # mass of electron    in erg
m_p_erg = (cst.m_p.cgs).to(u.erg, u.mass_energy())              # mass of proton      in erg
#m_AQN_GeV = 1 * u.g * g_to_GeV

B = 10**25                                                  # Baryon charge number
E_ann_GeV = 2 * u.GeV                                       # energy liberated by proton annihilation
f  = 1/10                                                   # factor to account for probability of reflection
g  = 1/10                                                   # (1-g) of total annihilation energy is thermalized     
Dv = 0.00013835783 * u.dimensionless_unscaled               # speed of nugget through visible matter
Dv = 10**-3 * u.dimensionless_unscaled

# n_bar = 1 * 1/u.cm**3
# n_AQN = 1.67*10**-24 * 1/u.cm**3
#  ^^ I don't think T_AQN depends on R_AQN or n_AQN



# m_AQN_kg = 0.23*u.kg
# R_AQN = calc_R_AQN(m_AQN_kg)

# R_AQN = 10**(-5) * u.cm
# m_AQN = 1 * u.g * g_to_GeV



