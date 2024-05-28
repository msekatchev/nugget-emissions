import numpy as np
from astropy import constants as cst
from astropy import units as u
print("Loaded survey parameters script")

# define bandwidth

# ############### GALEX #########################################
# Source: Henry 2015
wavel_min = 1350 * u.Angstrom
wavel_max = 1750 * u.Angstrom
f_max_hz = (cst.c.cgs / wavel_min.to(u.cm)).to(u.Hz)
f_min_hz = (cst.c.cgs / wavel_max.to(u.cm)).to(u.Hz)
skymap_units =  1*u.photon / u.cm**2 / u.s / u.Angstrom

# from erg/s/Hz/cm2 to photons/s/A/cm2
def convert_to_skymap_units(F_erg_hz_cm2,nu):
    erg_hz_cm2 = 1*u.erg/u.s/u.Hz/u.cm**2
    w = nu.to(u.AA, equivalencies=u.spectral())
    C = (erg_hz_cm2).to(skymap_units, u.spectral_density(w)) / erg_hz_cm2 * 2*np.pi
    #                                                                      ^^^^^^^ this comes from using hbar instead of h in the conversion.
    return F_erg_hz_cm2 * C
#################################################################


################ WMAP K-Band ###################################
# Source: https://lambda.gsfc.nasa.gov/product/wmap/current/
# f_centre_GHz = 23 * u.GHz
# f_min_hz = (f_centre_GHz - 5.5/2 * u.GHz).to(u.Hz)
# f_max_hz = (f_centre_GHz + 5.5/2 * u.GHz).to(u.Hz)
# skymap_units = u.Jy

# from erg/s/Hz/cm2 to mK
# def convert_to_skymap_units(F_erg_hz_cm2,nu):
#     erg_hz_cm2 = 1*u.erg/u.s/u.Hz/u.cm**2
#     C = erg_hz_cm2.to(skymap_units)
#     return (F_erg_hz_cm2 * C / erg_hz_cm2).to(u.mK, equivalencies = u.brightness_temperature(nu))
#################################################################



