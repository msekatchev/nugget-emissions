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

from time import time as tt




band_min, band_max = 1300*u.AA, 1700*u.AA

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


sigma_v = 156 * u.km/u.s
v_b = 180 * u.km/u.s

m_aqn_kg = 16.7/1000 * u.kg

quant = {
    'dark_mat': np.array([0.3,0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01,0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0,0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e6,1e4]) * u.K, 
    'dv_ioni':  np.array([220,100]) * u.km/u.s, 
    'dv_neut':  np.array([0,0]) * u.km/u.s,
}

N = 5

quant = {
    'dark_mat': np.random.uniform(0.2, 0.4, size=N) * u.GeV / u.cm**3 * GeV_to_g,
    'ioni_gas': np.random.uniform(0.005, 0.02, size=N) * 1 / u.cm**3,
    'neut_gas': np.random.uniform(0, 0.01, size=N) * 1 / u.cm**3,
    'temp_ion': np.random.uniform(1e3, 1e5, size=N) * u.K,
    'dv_ioni': np.random.uniform(50, 300, size=N) * u.km / u.s,
    'dv_neut': np.random.uniform(0, 50, size=N) * u.km / u.s,
}

# quant = {
#     'dark_mat': np.array([0.3,0.3,0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01,0.01,0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0,0,0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e6,1e4,1e3]) * u.K, 
#     'dv_ioni':  np.array([220,100,120]) * u.km/u.s, 
#     'dv_neut':  np.array([0,0,0]) * u.km/u.s,
# }

# quant = {
#     'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
#     'ioni_gas': np.array([0.01]) * 1/u.cm**3,
#     'neut_gas': np.array([0]) * 1/u.cm**3, 
#     'temp_ion': np.array([1e6]) * u.K, 
#     'dv_ioni':  np.array([120]) * u.km/u.s, 
#     'dv_neut':  np.array([0]) * u.km/u.s,
# }



# res = compute_epsilon_ionized_bandwidth_2(quant, m_aqn_kg, 1300*u.AA, 1700*u.AA, adjust_T_gas=False)
# # print(res)
# print(res["aqn_emit"]*(0.6*u.kpc).to(u.cm)/(4*np.pi))




# x0 = np.array([10,1])
# kT = ((2*np.pi*cst.hbar*cst.c)/(x0*lambda0)).to(u.J)
# lamb = 100
# x = ((2*np.pi*cst.hbar*cst.c)/(kT*lamb*u.AA)).to(u.dimensionless_unscaled)

# print(H())

def compute_epsilon_velocity_integral(quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b):
                # !!!!! quad_vec fails when adjust_T_gas = True
    result, error = quad_vec(epsilon_velocity_integrand, 0.1, 1*(sigma_v.value+v_b.value), 
        args=(quant, sigma_v.value, v_b.value, m_aqn_kg, band_min, band_max, adjust_T_gas))

    return result * epsilon_units/u.sr

# def compute_epsilon_velocity_integral(quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b):
#                 # !!!!! quad_vec fails when adjust_T_gas = True
#     batches = split_quant(quant, n_splits)

#     # result, error = quad(epsilon_velocity_integrand, 0.1, 1*(sigma_v.value+v_b.value), 
#     #     args=(quant, sigma_v.value, v_b.value, m_aqn_kg, band_min, band_max, adjust_T_gas))

#     batch_results = {}
#     for i in range(n_splits):
#     	batch_list = {}
#     	for j in range(len(batches[i])):
#     		print(batches)
#     		batch_list[j] = quad(epsilon_velocity_integrand, 0.1, 1*(sigma_v.value+v_b.value), 
#         args=(batches[i][j], sigma_v.value, v_b.value, m_aqn_kg, band_min, band_max, adjust_T_gas))[0]

#     	batch_results[i] = batch_list

#     return batch_results * epsilon_units/u.sr

# sigma_v, v_b = 50*u.km/u.s, 50*u.km/u.s


# t=tt()
# res = compute_epsilon_velocity_integral(quant, m_aqn_kg, 1300*u.AA, 1700*u.AA, True, sigma_v, v_b)
# print(res*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# ttt(t,"2")

# N = 5

# quant = {
#     'dark_mat': np.random.uniform(0.2, 0.4, size=N) * u.GeV / u.cm**3 * GeV_to_g,
#     'ioni_gas': np.random.uniform(0.005, 0.02, size=N) * 1 / u.cm**3,
#     'neut_gas': np.random.uniform(0, 0.01, size=N) * 1 / u.cm**3,
#     'temp_ion': np.random.uniform(1e3, 1e5, size=N) * u.K,
#     'dv_ioni': np.random.uniform(50, 300, size=N) * u.km / u.s,
#     'dv_neut': np.random.uniform(0, 50, size=N) * u.km / u.s,
# }

# enforce_units(quant)


# t=tt()
# res = compute_epsilon(quant, m_aqn_kg, 1300*u.AA, 1700*u.AA, True, sigma_v, v_b, False)
# ttt(t,"1")
# print(res)
# print("------------------------------------------------")



# t=tt()
# res = compute_epsilon(quant, m_aqn_kg, 1300*u.AA, 1700*u.AA, True, sigma_v, v_b, True)
# ttt(t,"2")
# print(res)
# print("------------------------------------------------")

# print(res*(0.6*u.kpc).to(u.cm)/(4*np.pi))
# ttt(t,"2")



# batches = split_quant(quant, 10)

# print(batches[0])
# print("---------------")
# print(batches[1])
# print("---------------")
# print(batches[2])
# print("---------------")
# print(batches[3])
# print("---------------")

n_splits = 3
# compute_epsilon_velocity_integral(quant, m_aqn_kg, band_min, band_max, True, sigma_v, v_b)


# from joblib import Parallel, delayed
# import numpy as np

# def process_batch(batch_quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b):
#     """Process a batch of cubes."""
#     results = []
#     for i in range(len(batch_quant["dark_mat"])):  # Process each cube in the batch
#         cube_quant = {key: val[i:i + 1] for key, val in batch_quant.items()}
#         result = compute_epsilon_velocity_integral(
#             cube_quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b
#         )
#         results.append(result)
#     return np.array(results)

# def parallel_compute(quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b, n_jobs=-1):
#     """Parallelize computation over batches of cubes."""
#     # Determine the number of workers (batches) to use
#     from multiprocessing import cpu_count
#     n_workers = n_jobs if n_jobs > 0 else cpu_count()

#     # Split quant into batches
#     quant_batches = split_quant(quant, n_workers)

#     # Process each batch in parallel
#     batch_results = Parallel(n_jobs=n_jobs)(
#         delayed(process_batch)(batch, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b)
#         for batch in quant_batches
#     )
    
#     # Combine results from all batches
#     # aqn_emit = np.concatenate(batch_results, axis=0)
#     # aqn_emit = np.vstack(batch_results)
#     print(batch_results)
#     print(len(quant["dark_mat"]))
#     aqn_emit = [batch_results[i][0][0] for i in range(n_workers)]
#     return 

# aqn_emit = parallel_compute(quant, m_aqn_kg, 1300*u.AA, 1700*u.AA, False, sigma_v, v_b)


# v_array = np.linspace(0.01, 1*(sigma_v.value+v_b.value), 500) * u.km/u.s
# res_array = {}# np.zeros(len(v_array)) * photon_units
# for i in range(len(v_array)):
# 	t = tt()
# 	res_array[i] = epsilon_velocity_integrand(v_array[i].value, quant, sigma_v.value, v_b.value, m_aqn_kg, 1300*u.AA, 1700*u.AA, True)*epsilon_units*(0.6*u.kpc).to(u.cm)/(4*np.pi)/u.sr
# 	ttt(t, 1)
# print(res_array)
# res_list = [res_array[i][0].value for i in range(500)]


# plt.scatter(v_array, res_list)
# plt.show()


# res = quad(f_maxbolt, 0, 10000)
# print(res)

# x_arr = np.linspace(0.001,1000,1000)
# res = f_maxbolt(x_arr)
# plt.plot(x_arr, res)
# plt.show()
# 


quant = {
    'dark_mat': np.array([0.3]) * u.GeV/u.cm**3 * GeV_to_g,
    'ioni_gas': np.array([0.01]) * 1/u.cm**3,
    'neut_gas': np.array([0]) * 1/u.cm**3, 
    'temp_ion': np.array([1e4]) * u.K, 
    'dv_ioni':  np.array([220]) * u.km/u.s, 
    'dv_neut':  np.array([0]) * u.km/u.s,
}
enforce_units(quant)

m_aqn_kg = 16.7/1000 * u.kg

print("sigma_v = 50, v_b = 50, Phi = 40.22")
# print(" --> ", compute_epsilon_velocity_integral(quant, m_aqn_kg, band_min, band_max, 
#     True, 50*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("------------------------------------------------------")

print("sigma_v = 50, v_b = 50, Phi = 40.22")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 50*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))
print(" --> ", compute_epsilon_no_bandwidth(quant, m_aqn_kg, 1500*u.AA, 
    True, 50*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))

print("sigma_v = 100, v_b = 50, Phi = 7.524")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 100*u.km/u.s, 50*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 110, v_b = 180, Phi = 1.702")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 110*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 156, v_b = 180, Phi = 1.167")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 156*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 50, v_b = 180, Phi = 0.1312")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 50*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 500, v_b = 180, Phi = 0.06485")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 500*u.km/u.s, 180*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 110, v_b = 500, Phi = 2.29831e-4")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 110*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))


print("sigma_v = 500, v_b = 500, Phi = 0.04198")
print(" --> ", compute_epsilon(quant, m_aqn_kg, band_min, band_max, 
    True, 500*u.km/u.s, 500*u.km/u.s)*(0.6*u.kpc).to(u.cm)/(4*np.pi))

print("------------------------------------------------------")










