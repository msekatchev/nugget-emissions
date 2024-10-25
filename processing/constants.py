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
kg_to_eV = g_to_GeV.to(u.eV/u.kg)

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

photon_units =  1*u.photon / u.cm**2 / u.s / u.Angstrom / u.sr
erg_hz_cm2   = 1*u.erg/u.s/u.Hz/u.cm**2
epsilon_units = 1*u.photon / u.cm**3 / u.s / u.Angstrom
epsilon_to_photon = (0.6*u.kpc).to(u.cm)/(4*np.pi)

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

# This is only added for the /sr dependence of the units
dOmega = 1*u.sr

# n_bar = 1 * 1/u.cm**3
# n_AQN = 1.67*10**-24 * 1/u.cm**3
#  ^^ I don't think T_AQN depends on R_AQN or n_AQN



# m_AQN_kg = 0.23*u.kg
# R_AQN = calc_R_AQN(m_AQN_kg)

# R_AQN = 10**(-5) * u.cm
# m_AQN = 1 * u.g * g_to_GeV

def enforce_units(quant):
    quant["dark_mat"] = quant["dark_mat"].to(u.kg/u.m**3)
    quant["ioni_gas"] = quant["ioni_gas"].to(1/u.cm**3)
    quant["neut_gas"] = quant["neut_gas"].to(1/u.cm**3)
    if quant["temp_ion"].unit == u.K:
        quant["temp_ion"] = quant["temp_ion"] * K_to_eV
    try: 
        if quant["dv_ioni"].unit != u.dimensionless_unscaled:
            quant["dv_ioni"] = (quant["dv_ioni"] / cst.c).to(u.dimensionless_unscaled)
    except:
        quant["dv_ioni"] = quant["dv_ioni"] * u.dimensionless_unscaled
    if quant["dv_neut"].unit != u.dimensionless_unscaled:
        quant["dv_neut"] = (quant["dv_neut"] / cst.c).to(u.dimensionless_unscaled)


