import numpy as np
from astropy import constants as cst
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import nquad
import scipy
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
study("quad")
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








# # WORK WITH A SINGLE INTEGRAL:
# x0 = 0.0005
# lambda0 = 1500 * u.AA

# x_min = ((2*np.pi*cst.hbar*cst.c)/(kT*1800*u.AA)).to(u.dimensionless_unscaled)
# x_max = ((2*np.pi*cst.hbar*cst.c)/(kT*1300*u.AA)).to(u.dimensionless_unscaled)



# true_value = 1.00506
# print(integral, true_value, (integral/true_value-1)*100,"%")

# PRINT A BUNCH OF VALUES FOR XUNYU










