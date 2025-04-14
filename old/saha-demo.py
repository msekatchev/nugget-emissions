import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as cst
from astropy import units as u


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

J_to_eV = 1/cst.e.si.value * u.eV / u.J
m_to_AA = 1e10 * u.AA / u.m
log2 = np.log(2)

m_e_eV  = (cst.m_e.cgs*cst.c.cgs**2).value * u.erg * erg_to_eV  # mass of electron    in eV



# Define the Saha equation for Hydrogen ionization fraction
def saha_ionization_fraction(T):
    """ Returns the ionization fraction of hydrogen at temperature T (in Kelvin). """
    kT = (cst.k_B * T).to(u.eV)  # Convert thermal energy to eV
    chi_H = 13.6 * u.eV  # Ionization energy of hydrogen
    g_ion = 2  # Statistical weight of ionized hydrogen (proton)
    g_neutral = 2  # Statistical weight of neutral hydrogen
    ne = 0.05 * u.cm**-3  # Electron number density (assumed)
    
    saha_ratio = ((2 * np.pi * cst.m_e * kT / (cst.h**2))**(3/2) *
                  (g_ion / g_neutral) * np.exp(-chi_H / kT) / ne).decompose()
    print(saha_ratio)
    return saha_ratio / (1 + saha_ratio)  # Ionization fraction

# Temperature range
# T_range = np.logspace(3, 5, 1000) * u.K
T_range = np.linspace(0,1e4,1000) * u.K
ionization_fraction = saha_ionization_fraction(T_range)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(T_range, ionization_fraction, label="Ionization Fraction")
# plt.axvline(3e4, color="r", linestyle="--", label=r"$3 \times 10^4$ K")
print(2.6*u.eV*eV_to_K)
# plt.axvline((13.6*u.eV*eV_to_K).value, color="r", linestyle="--", 
  # label="Ionization Energy of Hydrogen")
# plt.xscale("log")
plt.xlabel("Temperature (K)")
plt.ylabel("Ionization Fraction")
plt.title("Saha Equation: Hydrogen Ionization vs. Temperature")
plt.legend()
plt.grid(True)
plt.show()
