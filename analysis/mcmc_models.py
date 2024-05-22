import gizmo_analysis as gizmo  # rename these packages for brevity
import utilities as ut  # rename these packages for brevity

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from astropy import units as u
from astropy import constants as cst

# from aqn import *
# from constants import *
import pickle
import scipy.stats as stats

from scipy.stats import bootstrap

import emcee
from multiprocessing import Pool,cpu_count
import corner

# 2. likelihood
def compute_inv_cov_mat(errors, num_bins):
    cov_mat = np.zeros([num_bins, num_bins])
    for i in range(num_bins):
        cov_mat[i,i] = errors[i].value**2
    inv_cov = np.linalg.inv(cov_mat)

    return inv_cov

def lnlike(parameters, x, observation, theory_function, inv_cov):
    # vh, r0 = parameters
    theory = theory_function(x, parameters)

    difference = theory - observation

    lnlike = -0.5*float(difference @ inv_cov @ difference)
    return lnlike

# 3. numerator of Bayes' formula
def lnprob(parameters, x, observation, theory_function, inv_cov):
    # priors
    lgprior = lnprior(parameters)
    if not np.isfinite(lgprior): return -np.inf
    return lgprior + lnlike(parameters, x, observation, theory_function, inv_cov)

################################################################### MAIN FITTING FUNCTION ###################################################################################
class fit:
    def __init__(self, quant, masses, distances, cube_length=20, voxel_resolution=2**9):
        self.quant, self.masses, self.distances, self.cube_length, self.voxel_resolution = quant, masses, distances, cube_length, voxel_resolution

    def set_fit_parameters(self, max_distance_kpc, num_bins, distance_bins):
        self.max_distance_kpc, self.num_bins, self.distance_bins = max_distance_kpc, num_bins, distance_bins

    def density(data, distance_bins):       
        masses, distances = data[0], data[1]
        
        m, R, bin_numbers = stats.binned_statistic(distances,
                          masses.to(u.kg),
                          statistic='sum',
                          bins=distance_bins)
        R_inner, R_outer = R[0:-1]*u.kpc, R[1:]*u.kpc
        
        V_outer = 4/3 * np.pi * (R_outer).to(u.m)**3
        V_inner = 4/3 * np.pi * (R_inner).to(u.m)**3
    
        density      = m / (V_outer-V_inner)
        
        return density

    def velocity(data, distance_bins):       
        quant, distances = data[0], data[1]
        
        m, R, bin_numbers = stats.binned_statistic(distances,
                          quant.to(u.m/u.s),
                          statistic='mean',
                          bins=distance_bins)
        
        return m

    def temperature(data, distance_bins):       
        quant, distances = data[0], data[1]
        
        m, R, bin_numbers = stats.binned_statistic(distances,
                          quant.to(u.K),
                          statistic='mean',
                          bins=distance_bins)
        
        return m

    def density_cum(data, distance_bins):
        masses, distances = data[0], data[1]

        masses_cum = np.array([(np.sum(masses[distances < distance_bins[i]])).to(u.kg).value for i in range(1,
                      len(distance_bins))]) * u.kg
        
        return masses_cum

    def generate_density(self):
        masses, distances, distance_bins = self.masses, self.distances, self.distance_bins
        self.density = fit.density([masses, distances], distance_bins)

    def generate_density_cum(self):
        masses, distances, distance_bins = self.masses, self.distances, self.distance_bins
        self.masses_cum = fit.density_cum([masses, distances], distance_bins)

    def my_bootstrap(self, bsfunction, fraction, num_samples):
        distance_bins = self.distance_bins
        length = len(self.masses)
        samples_indexes = np.random.choice(length, size=(num_samples, round(fraction * length)))

        data = [self.masses, self.distances]
        
        res = np.array([bsfunction([data[0][samples_indexes[i,:]], data[1][samples_indexes[i,:]]], distance_bins) for i in range(num_samples)])
        
        return res*1/fraction, (1/fraction)**2 * np.cov(res.T)     

    def my_bootstrap_vel(self, bsfunction, fraction, num_samples):
        distance_bins = self.distance_bins
        length = len(self.distances)
        samples_indexes = np.random.choice(length, size=(num_samples, round(fraction * length)))

        data = [self.masses, self.distances]
        
        res = np.array([bsfunction([data[0][samples_indexes[i,:]], data[1][samples_indexes[i,:]]], distance_bins) for i in range(num_samples)])
        
        return res, np.cov(res.T)   
    
    def plot_covariance_matrix(self, quantity, sampled_quantity, cov_matrix):
        distance_bins, name = self.distance_bins, self.quant
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300)
       
        # Plot for the first column
        ax = axes[0]
        ax.set_title(f"True $\\rho$ - Mean Sampled $\\rho$")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")
        
        mass_diff = quantity.value - np.mean(sampled_quantity, axis=0)
    
        ax.scatter(distance_bins[1:], mass_diff)
        ax.axhline(0, color="r", linestyle="--")
        for i in range(len(distance_bins) - 1):
            ax.plot([distance_bins[i+1].value, distance_bins[i+1].value], [0, mass_diff[i]], color='blue')
    
        ax = axes[1]
        ax.set_title(f"Correlations")
        
        x, y = np.meshgrid(distance_bins.value, distance_bins.value)
        res = ax.pcolor(x, y, np.corrcoef(np.transpose(sampled_quantity)))
    
        ax.invert_yaxis()
        plt.colorbar(res)
    
        
        # Plot for the second column
        ax = axes[2]
        ax.set_title(f"inv($\Sigma$)$\cdot\Sigma-1$")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Distance")
    
        x, y = np.meshgrid(distance_bins.value, distance_bins.value)
        res = ax.pcolor(x, y, np.linalg.inv(cov_matrix) @ cov_matrix - np.diag(np.ones(len(distance_bins)-1)))
        ax.invert_yaxis()
        plt.colorbar(res)    
        
        plt.tight_layout()   
        
        plt.savefig("../visuals/cov-degeneracy-result-"+name+".png", bbox_inches='tight')
        plt.savefig("../visuals/cov-degeneracy-result-"+name+".svg", bbox_inches='tight')
        plt.show()

    def run_mcmc(self, quantity, model_name, 
                 model_function, lnprior_function, lnprob_function, 
                 parameter_names, initial_guess, nwalkers, nsteps, burnout):  

        distance_bins, quant = self.distance_bins, self.quant
        self.model_name, self.initial_guess, self.parameter_names = model_name, initial_guess, parameter_names
        self.nwalkers, self.nsteps, self.burnout = nwalkers, nsteps, burnout
        
        ndim = len(initial_guess)
        starting_pts = [initial_guess + 0.1*np.random.randn(ndim) for i in range(nwalkers)] # initial points for each walkers
        with Pool() as pool: # parallelization, remove this if you only want 1 cpu
            sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob_function,pool=pool, args=(distance_bins[1:].value, quantity.value, model_function, self.inv_cov_matrix)) 
            sampler.run_mcmc(starting_pts,nsteps,progress=True) 
            # run your mcmc, you can monitor that via a progress bar if you have some package (that I forgot) installed
            # otherwise you need to set progress=False
    
        analysis_chain = sampler.flatchain[nwalkers*burnout:]
        # np.save('../data/FIRE/MCMC-chains/'+quant+"-"+model_name+".npy", sampler.flatchain) # here saves your sampled posterior chain files
        
        # RESULTS:
        self.posterior_means = np.average(np.array(analysis_chain),axis=0) # use posterior mean instead of max posterior, which is what mcmc bad at to find
        self.posterior_std = np.std(np.array(analysis_chain),axis=0)
        self.analysis_chain = analysis_chain
        
        toprint = 'The constraints are:\n'
        for i in range(ndim):
            # toprint = toprint+parameter_names[i]+'='+str(posterior_means[i])[:5]+'+-'+str(posterior_std[i])[:4]+"\n"
            toprint = toprint+parameter_names[i]+' = '+str(self.posterior_means[i])+' +- '+str(self.posterior_std[i])+"\n"
        print(toprint)

        self.plot_mcmc_results(sampler, quantity, model_function)

    def plot_mcmc_results(self, sampler, quantity, model_function):
        quant = self.quant
        model_name, initial_guess, parameter_names = self.model_name, self.initial_guess, self.parameter_names
        nwalkers, nsteps, burnout = self.nwalkers, self.nsteps, self.burnout
        distance_bins = self.distance_bins

        fig,ax = plt.subplots(len(initial_guess),1, dpi=300) # figsize=(len(initial_guess)*1.5, 6)
        res = [ax[i].plot(sampler.chain[:,:,i].T, '-', color='k', alpha=0.3, linewidth=0.5) for i in range(len(initial_guess))]
        res = [ax[i].plot(np.arange(0,len(sampler.flatchain)/nwalkers,1/nwalkers), sampler.flatchain[0:][:,i], 
                          '-', color='red', alpha=0.1, linewidth=0.5) for i in range(len(initial_guess))]
        res = ax[len(initial_guess)-1].set_xlabel("step",size=12)
        res = [ax[i].set_ylabel(parameter_names[i], size=12) for i in range(len(initial_guess))]
        res = [ax[i].set_xticks([]) for i in range(len(initial_guess)-1)]
        res = [ax[i].axvline(nwalkers*burnout/nwalkers, color="red") for i in range(len(initial_guess))]
        res = [ax[i].axhline(initial_guess[i]) for i in range(len(initial_guess))]
        plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-walkers.png", bbox_inches='tight')
        plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-walkers.svg", bbox_inches='tight')
        plt.show()
        # plt.figure(dpi=150)
        # plt.errorbar(R_outer, rho_c, yerr=rho_c_error, marker=".", markersize=1, linestyle = 'none',color='blue', ecolor="gray", elinewidth=0.7)
        # plt.plot(R_outer, rho_c_burkert_2(R_outer.value, posterior_means))
        # # plt.plot(R_outer, rho_c_burkert_quad(R_outer.value, vh*10, r0*1.4))
        # plt.title("Cumulative Density Distribution")
        # plt.xlabel("R [kpc]", size=20)
        # plt.ylabel(r'$\rho^c_{DM}$'+"  [kg$\cdot$m$^{-3}$]", size = 20)
        # plt.yscale('log')
        # plt.show()
    
        plt.figure(dpi=300)
        plt.errorbar(distance_bins[1:], quantity, yerr=np.std(self.sampled_density, axis=0)*u.kg/u.m**3, marker=".", markersize=1, linestyle = 'none',color='blue', ecolor="gray", elinewidth=0.7)
        plt.plot(distance_bins[1:], model_function(distance_bins[1:].value, self.posterior_means))
        # plt.plot(R_outer, rho_c_burkert_quad(R_outer.value, vh*10, r0*1.4))
        plt.title("Density Distribution, "+model_name)
        plt.xlabel("R [kpc]", size=20)
        plt.ylabel(r'$\rho_{DM}$'+"  [kg$\cdot$m$^{-3}$]", size = 20)
        plt.yscale('log')
        plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-fit.png", bbox_inches='tight')
        plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-fit.svg", bbox_inches='tight')
        plt.show()
        
        corner_fig = plt.figure(dpi=400)
        corner.corner(self.analysis_chain, labels=parameter_names, truths=self.posterior_means, fig=corner_fig)
        plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-corner.png", bbox_inches='tight')
        plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-corner.svg", bbox_inches='tight')
        plt.show()


########### Burkert #############################
# 0. distribution model
def rho_burkert(R, parameters):
    # initial guesses: [vh, r0] = [4.14, 7.8] # [km/(100s), kpc]
    
    vh, r0 = parameters
    
    R_kpc   = R  * u.kpc

    vh_km_s = vh * 100 * u.km / u.s
    r0_kpc  = r0 * u.kpc
    
    rho0_kg_m3 = (vh_km_s**2 / (4*np.pi * r0_kpc**2 * cst.G)).to(u.kg/u.m**3)
    
    return (rho0_kg_m3 * r0_kpc**3 / ( (R_kpc+r0_kpc)*((R_kpc)**2 + r0_kpc**2) )).value # kg/m^3

def rho_burkert_cum(R, parameters):
    return cumtrapz(2 * np.pi * ((R*u.kpc).to(u.m))**2 * rho_burkert(R, parameters), R, initial=0)

# 1. prior
def lnprior_burkert(parameters):
    vh, r0 = parameters
    if vh <= 0 or r0 <= 0 or vh*1/(3*10**5)>=1:
        return -np.inf
    return 0

def lnprob_burkert(parameters, x, observation, theory_function, inv_cov):
    # priors
    lgprior = lnprior_burkert(parameters)
    if not np.isfinite(lgprior): return -np.inf
    return lgprior + lnlike(parameters, x, observation, theory_function, inv_cov)
#####################################################################

########### Generalized Einasto Profile #############################
def rho_gEinasto(R, parameters):
    rho_s, r_s, alpha, gamma = parameters
    
    R_kpc   = R  * u.kpc
    rho_s_msol_kpc3 = rho_s * u.solMass / u.kpc**3 * 1e7
    r_s_kpc = r_s * u.kpc
    
    rho_s_kg_m3 = (rho_s_msol_kpc3).to(u.kg/u.m**3)

    return (rho_s_kg_m3 * (R_kpc/r_s_kpc)**(-gamma) * np.exp((-2/alpha)*((R_kpc/r_s_kpc)**alpha - 1))).value  

def lnprior_gEinasto(parameters):
    rho_s, r_s, alpha, gamma = parameters
    if rho_s < 0 or r_s < 0 or r_s > 200 or alpha < 0 or alpha > 5 or gamma < -5 or gamma > 5:
        return -np.inf
    return 0

def lnprob_gEinasto(parameters, x, observation, theory_function):
    # priors
    lgprior = lnprior_gEinasto(parameters)
    if not np.isfinite(lgprior): return -np.inf
    return lgprior + lnlike(parameters, x, observation, theory_function)
#####################################################################

########### Einasto Profile #########################################
def rho_Einasto(R, parameters):
    rho_s, r_s, alpha = parameters
    return rho_gEinasto(R, [rho_s, r_s, alpha, 0])

def lnprior_Einasto(parameters):
    rho_s, r_s, alpha = parameters
    if rho_s < 0 or r_s < 0 or r_s > 200 or alpha < 0 or alpha > 5:
        return -np.inf
    return 0

def lnprob_Einasto(parameters, x, observation, theory_function):
    # priors
    lgprior = lnprior_Einasto(parameters)
    if not np.isfinite(lgprior): return -np.inf
    return lgprior + lnlike(parameters, x, observation, theory_function)
#####################################################################

########### Generalized NFW Profile #################################
def rho_gNFW(R, parameters):
    rho_s, r_s, alpha, beta, gamma = parameters
        
    R_kpc   = R  * u.kpc
    rho_s_msol_kpc3 = rho_s * u.solMass / u.kpc**3 * 1e7
    r_s_kpc = r_s * u.kpc
    
    rho_s_kg_m3 = (rho_s_msol_kpc3).to(u.kg/u.m**3)
    
    return (( (2**((beta - gamma)/alpha)) * rho_s_kg_m3 ) / ( (R_kpc/r_s_kpc)**gamma * (1+(R_kpc/r_s_kpc)**alpha)**((beta - gamma)/alpha)  )).value

def lnprior_gNFW(parameters):
    rho_s, r_s, alpha, beta, gamma = parameters
    if rho_s <= 0 or r_s <= 0 or r_s >= 200 or alpha <= 0 or alpha >= 5 or beta <= 0 or beta >= 10 or gamma <= 0 or gamma >= 5:
        return -np.inf
    return 0

def lnprob_gNFW(parameters, x, observation, theory_function):
    # priors
    lgprior = lnprior_gNFW(parameters)
    
    # print(lgprior, "_")
    
    if (not np.isfinite(lgprior)) or (np.isnan(lgprior)): return -np.inf
    return lgprior + lnlike(parameters, x, observation, theory_function)
#####################################################################


########### NFW Profile #############################################
def rho_NFW(R, parameters):
    # initial guesses: [rho_0h, r_h] = [1.0,1.9] # [100solMass/pc^3, kpc/10]
    
    rho_0h, r_h = parameters    
    
    R_kpc   = R  * u.kpc
    
    rho_0h_mol_pc  = rho_0h * u.solMass / u.pc**3 / 100
    r_h_kpc     = r_h * u.kpc * 10

    return (rho_0h_mol_pc.to(u.kg/u.m**3) / ((R_kpc/r_h_kpc)*(1+R_kpc/r_h_kpc)**2))

def rho_NFW_cum(R, parameters):
    # initial guesses: [rho_0h, r_h] = [1.0,1.9] # [100solMass/pc^3, kpc/10
    def integrate_func(R, parameters):
        return 2 * np.pi * ((R*u.kpc).to(u.m))**2 * (R*u.kpc).to(u.m) * rho_NFW(R, parameters)
    return cumtrapz(integrate_func(R,parameters), R, initial=0).value

def lnprior_NFW(parameters):
    rho_0h, r_h = parameters
    if rho_0h <= 0 or r_h <= 0 or r_h >= 200:
        return -np.inf
    return 0

def lnprob_NFW(parameters, x, observation, theory_function):
    # priors
    lgprior = lnprior_NFW(parameters)
    
    if not np.isfinite(lgprior): return -np.inf
    return lgprior + lnlike(parameters, x, observation, theory_function)
#####################################################################

########### dark matter profile (generalized NFW) ###################
def rho_dark_mat(R, parameters):
    rho_s, r_s, alpha, beta, gamma = parameters
        
    R_kpc   = R  * u.kpc
    rho_s_msol_kpc3 = rho_s * u.solMass / u.kpc**3 * 1e7
    r_s_kpc = r_s * u.kpc
    
    rho_s_kg_m3 = (rho_s_msol_kpc3).to(u.kg/u.m**3)
    
    return (( (2**((beta - gamma)/alpha)) * rho_s_kg_m3 ) / ( (R_kpc/r_s_kpc)**gamma * (1+(R_kpc/r_s_kpc)**alpha)**((beta - gamma)/alpha)  )).value
#####################################################################

########### ionized gas #############################################
def rho_ioni_gas(x, parameters):
    a1, b1, c1 = parameters[0]*1e-23, parameters[1]*10, parameters[2] 
    a2, b2, c2 = parameters[3]*1e-23, parameters[4], parameters[5]
    a3         = parameters[6]*1e-24

    # a1, b1, c1 = 6.2e-23, 1.2, 10
    # a2, b2, c2 = 1.3e-23, 1.3, 1
    # a3         = 0.15e-23
    
    ret = a1 / (1+(x/b1)**2)**c1 + \
          a2 / (1+(x/b2)**2)**c2 + \
          a3
    
    return ret

def lnprior_ioni_gas(parameters):
    a1,b1,c1,a2,b2,c2,a3 = parameters
    a1,a2,a3 = a1*1e-23,a2*1e-23,a3*1e-24

    # print(parameters,"\n\n")
    
    if np.sum(np.array(parameters) <= 0) >= 1 or np.sum(np.array(parameters) > 30) >= 1: #or b1>=10 or b2>=10:
        return -np.inf
    return 0

def lnprob_ioni_gas(parameters, x, observation, theory_function, inv_cov):
    # priors
    lgprior = lnprior_ioni_gas(parameters)
    if not np.isfinite(lgprior): return -np.inf
    return lgprior + lnlike(parameters, x, observation, theory_function, inv_cov)
#####################################################################

########### neutral gas #############################################
def rho_neut_gas(x, parameters):
    a1, b1, c1 = parameters[0]*1e-21, parameters[1], parameters[2] 
    a2, b2, c2 = parameters[3]*1e-21, parameters[4], parameters[5]
    a3         = parameters[6]*1e-24

    ret = a1 / (1+(x/b1)**2)**c1 + \
          a2 / (1+(x/b2)**2)**c2 + \
          a3
    
    return ret

def lnprior_neut_gas(parameters):
    a1,b1,c1,a2,b2,c2,a3 = parameters
    a1,a2,a3 = a1*1e-23,a2*1e-23,a3*1e-24

    # print(parameters,"\n\n")
    
    if np.sum(np.array(parameters) <= 0) >= 1 or np.sum(np.array(parameters) > 50) >= 1: #or b1>=10 or b2>=10:
        return -np.inf
    return 0

def lnprob_neut_gas(parameters, x, observation, theory_function, inv_cov):
    # priors
    lgprior = lnprior_ioni_gas(parameters)
    if not np.isfinite(lgprior): return -np.inf
    return lgprior + lnlike(parameters, x, observation, theory_function, inv_cov)
#####################################################################

########### dv ionized #############################################
def rho_dv_ioni(x, parameters):
    a1, b1, c1 = parameters[0]*1e5, parameters[1]*10, parameters[2]*10 
    a2, b2, c2 = parameters[3]*1e5, parameters[4], parameters[5]
    a3         = parameters[6]*1e5
    
    ret = a1 / (1+(x/b1)**2)**c1 + \
          a2 / (1+(x/b2)**2)**c2 + \
          a3
    
    return ret
#####################################################################

########### dv neutral #############################################
def rho_dv_neut(x, parameters):
    a1, b1, c1 = parameters[0]*1e5, parameters[1]*1e2, parameters[2]*1e2 
    a2         = parameters[3]*1e5
    
    ret = a1 / (1+(x/b1)**2)**c1 + \
          a2
    
    return ret
#####################################################################

########### ionized gas temperature ################################
def rho_temp(x, parameters):
    a1, b1, c1 = parameters[0]*1e7, parameters[1], parameters[2]
    a2, d2     = parameters[3]*1e7, parameters[4]
    a3         = parameters[5]*1e7
    
    ret = a1 / (1+(x/b1)**2)**c1 + \
          a2 * np.tanh(x-d2) + a3
    
    return ret
#####################################################################

# def run_mcmc(distance_bins, quantity, quant, model_name, model_function, lnprior_function, lnprob_function, parameter_names, initial_guess, nwalkers, nsteps, burnout):  
    
#     ndim = len(initial_guess)
#     starting_pts = [initial_guess + 0.1*np.random.randn(ndim) for i in range(nwalkers)] # initial points for each walkers
#     with Pool() as pool: # parallelization, remove this if you only want 1 cpu
#         sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob_function,pool=pool, args=(distance_bins[1:].value, quantity.value, model_function)) 
#         sampler.run_mcmc(starting_pts,nsteps,progress=True) 
#         # run your mcmc, you can monitor that via a progress bar if you have some package (that I forgot) installed
#         # otherwise you need to set progress=False

#     analysis_chain = sampler.flatchain[nwalkers*burnout:]
#     # np.save('../data/FIRE/MCMC-chains/'+quant+"-"+model_name+".npy", sampler.flatchain) # here saves your sampled posterior chain files
    
#     # means
#     posterior_means = np.average(np.array(analysis_chain),axis=0) # use posterior mean instead of max posterior, which is what mcmc bad at to find
#     posterior_std = np.std(np.array(analysis_chain),axis=0)
#     toprint = 'The constraints are:\n'
#     for i in range(ndim):
#         # toprint = toprint+parameter_names[i]+'='+str(posterior_means[i])[:5]+'+-'+str(posterior_std[i])[:4]+"\n"
#         toprint = toprint+parameter_names[i]+' = '+str(posterior_means[i])+' +- '+str(posterior_std[i])+"\n"
#     print(toprint)
    

#     # plots
#     fig,ax = plt.subplots(len(initial_guess),1, dpi=300) # figsize=(len(initial_guess)*1.5, 6)
#     res = [ax[i].plot(sampler.chain[:,:,i].T, '-', color='k', alpha=0.3, linewidth=0.5) for i in range(len(initial_guess))]
#     res = [ax[i].plot(np.arange(0,len(sampler.flatchain)/nwalkers,1/nwalkers), sampler.flatchain[0:][:,i], 
#                       '-', color='red', alpha=0.1, linewidth=0.5) for i in range(len(initial_guess))]
#     res = ax[len(initial_guess)-1].set_xlabel("step",size=12)
#     res = [ax[i].set_ylabel(parameter_names[i], size=12) for i in range(len(initial_guess))]
#     res = [ax[i].set_xticks([]) for i in range(len(initial_guess)-1)]
#     res = [ax[i].axvline(nwalkers*burnout/nwalkers, color="red") for i in range(len(initial_guess))]
#     res = [ax[i].axhline(initial_guess[i]) for i in range(len(initial_guess))]
#     plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-walkers.png", bbox_inches='tight')
#     plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-walkers.svg", bbox_inches='tight')
#     plt.show()
#     # plt.figure(dpi=150)
#     # plt.errorbar(R_outer, rho_c, yerr=rho_c_error, marker=".", markersize=1, linestyle = 'none',color='blue', ecolor="gray", elinewidth=0.7)
#     # plt.plot(R_outer, rho_c_burkert_2(R_outer.value, posterior_means))
#     # # plt.plot(R_outer, rho_c_burkert_quad(R_outer.value, vh*10, r0*1.4))
#     # plt.title("Cumulative Density Distribution")
#     # plt.xlabel("R [kpc]", size=20)
#     # plt.ylabel(r'$\rho^c_{DM}$'+"  [kg$\cdot$m$^{-3}$]", size = 20)
#     # plt.yscale('log')
#     # plt.show()

#     plt.figure(dpi=300)
#     plt.errorbar(distance_bins[1:], quantity, yerr=np.std(sampled_densities, axis=0)*u.kg/u.m**3, marker=".", markersize=1, linestyle = 'none',color='blue', ecolor="gray", elinewidth=0.7)
#     plt.plot(distance_bins[1:], model_function(distance_bins[1:].value, posterior_means))
#     # plt.plot(R_outer, rho_c_burkert_quad(R_outer.value, vh*10, r0*1.4))
#     plt.title("Density Distribution, "+model_name)
#     plt.xlabel("R [kpc]", size=20)
#     plt.ylabel(r'$\rho_{DM}$'+"  [kg$\cdot$m$^{-3}$]", size = 20)
#     plt.yscale('log')
#     plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-fit.png", bbox_inches='tight')
#     plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-fit.svg", bbox_inches='tight')
#     plt.show()
    
#     corner_fig = plt.figure(dpi=400)
#     corner.corner(
#         analysis_chain, labels=parameter_names, truths=posterior_means, fig=corner_fig
#     )
#     plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-corner.png", bbox_inches='tight')
#     plt.savefig("../visuals/FIRE-MCMC-"+quant+"-"+model_name+"-corner.svg", bbox_inches='tight')
#     plt.show()
    
#     return analysis_chain, posterior_means
