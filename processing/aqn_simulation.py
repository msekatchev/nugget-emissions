import matplotlib.pyplot as plt

import numpy as np

from astropy import constants as cst
from astropy import units as u
import healpy as hp

from survey_parameters import *
from aqn import *
from constants import *

def simulate_signal(m_aqn_kg):
    sun_mw_distance_kpc = 8*u.kpc

    dm_model = "Burkert Profile"
    vm_model = "Gas Components"
    l_min = 0.6
    l_max = 50
    dl =0.5*u.kpc
    NSIDE = 2**5
    disp = True
    ionized_gas = True

    R_aqn_cm = calc_R_AQN(m_aqn_kg)
    
    # ########### NFW Profile #############################################
    if dm_model == "NFW Profile":
        rho_0h = 0.0106 * u.solMass / u.pc**3
        r_h = 19.0 * u.kpc

        def rho_halo(R_kpc):
            return rho_0h.to(u.kg/u.m**3) / ((R_kpc/r_h)*(1+R_kpc/r_h)**2)

        def n_halo(R_none):
            return (rho_halo(R_none*u.kpc) / m_aqn_kg).to(1/u.cm**3)
    # #####################################################################

    # # ########### constant density #########################################
    if dm_model == "Constant DM density":
        n_aqn_const_cm3 = 0.3/1e25 * u.cm**-3
        def n_halo(R_none):
            return np.ones(np.shape(R_none)) * n_aqn_const_cm3
    # # ######################################################################

    ########### Burkert Profile #########################################
    if dm_model == "Burkert Profile":
        # defaults:
        vh = 414 * u.km / u.s
        r0 = 7.8 * u.kpc
        
        # my changes:
        # [1,414,]
        vh = 26.566 * u.km / u.s
        r0 = 4.757 * u.kpc
        
        rho0 = (vh**2 / (4*np.pi * r0**2 * cst.G)).to(u.kg/u.m**3)

        def rho_halo(R_kpc):
            return rho0 * r0**3 / ( (R_kpc+r0)*((R_kpc)**2 + r0**2) )

        def n_halo(R_none):
            return (rho_halo(R_none*u.kpc) / m_aqn_kg).to(1/u.cm**3)
    #####################################################################

    if disp:
        plt.figure(dpi=100,figsize=(3,3))
        plt.plot(np.arange(l_min,l_max,0.1), n_halo(np.arange(l_min,l_max,0.1)))
        plt.xlabel("$R$ [kpc]")
        plt.ylabel("$n_{DM}$ [cm$^{-3}$]")
        plt.yscale("log")
        plt.title(dm_model, size=15)
        #plt.savefig(dm_model_name+".png", bbox_inches='tight')
        plt.show()

    ########################################################################


    ############# constant density ######################################### 
    if vm_model == "Constant VM density":
        n_cold_cm3 = 1.0e-1 * u.cm**-3
        def n_vm(R_none):
            return np.ones(np.shape(R_none)) * n_cold_cm3
    ########################################################################

    ############# gas components ######################################### 
    if vm_model == "Gas Components":
        [a1_c, b1_c, c1_c, 
         a2_c, b2_c, c2_c, 
         a3_c, b3_c, c3_c, 
         a4_c, b4_c, c4_c, 
         a5_c, b5_c, c5_c, 
         a6_c, b6_c, c6_c] = [30.04649796,  0.4079755,  3.77607598,  
                      1e3,3e-1,15,
                     2.6e4,5e-2,3,
                     0,0,0,
                     0,0,0,
                     0,0,0,]

        [a1_w, b1_w, c1_w, 
         a2_w, b2_w, c2_w, 
         a3_w, b3_w, c3_w, 
         a4_w, b4_w, c4_w, 
         a5_w, b5_w, c5_w, 
         a6_w, b6_w, c6_w] = [1.1e0,1e-1,0.22,
                      1.7e1,12e-1,110,
                      0.575e4,8.2e-2,13.7,
                     0,0,0,
                     0,0,0,
                     0,0,0,]

        [a1_wh, b1_wh, c1_wh, 
         a2_wh, b2_wh, c2_wh, 
         a3_wh, b3_wh, c3_wh, 
         a4_wh, b4_wh, c4_wh, 
         a5_wh, b5_wh, c5_wh, 
         a6_wh, b6_wh, c6_wh] = [6.1e1,4.4e-1,1.8,
                      9e2,9.2e-2,2.8,
                      1.5e4,7e-2,9,
                     0,0,0,
                     0,0,0,
                     0,0,0,]

        [a1_h, b1_h, c1_h, 
         a2_h, b2_h, c2_h, 
         a3_h, b3_h, c3_h, 
         a4_h, b4_h, c4_h, 
         a5_h, b5_h, c5_h, 
         a6_h, b6_h, c6_h] = [2e2,4e-1,9.5,
                      1e4,10e-2,6,
                      2.6e4,3e-2,3.1,
                      0,0,0,
                      0,0,0,
                      0,0,0,]

        from astropy.cosmology import WMAP7             # WMAP 7-year cosmology
        rho_crit_z05_cgs = WMAP7.critical_density(0.5)  # critical density at z = 0.5  
        rho_crit_z05_si = rho_crit_z05_cgs.to(u.kg / u.m**3)
        r_vir_kpc = 233 * u.kpc

        def rho_gas_component(R_kpc,gas_component):
            x = (R_kpc/r_vir_kpc).value
            if gas_component == "cold":
                return  (a1_c / (1+(x/b1_c)**2)**c1_c + \
                         a2_c / (1+(x/b2_c)**2)**c2_c + \
                         a3_c / (1+(x/b3_c)**2)**c3_c) * rho_crit_z05_si
            if gas_component == "warm":
                return  (a1_w / (1+(x/b1_w)**2)**c1_w + \
                         a2_w / (1+(x/b2_w)**2)**c2_w + \
                         a3_w / (1+(x/b3_w)**2)**c3_w) * rho_crit_z05_si
            if gas_component == "warm-hot":
                return  (a1_wh / (1+(x/b1_wh)**2)**c1_wh + \
                         a2_wh / (1+(x/b2_wh)**2)**c2_wh + \
                         a3_wh / (1+(x/b3_wh)**2)**c3_wh) * rho_crit_z05_si 
            if gas_component == "hot":
                return  (a1_h / (1+(x/b1_h)**2)**c1_h + \
                         a2_h / (1+(x/b2_h)**2)**c2_h + \
                         a3_h / (1+(x/b3_h)**2)**c3_h) * rho_crit_z05_si 

        def n_gas_component(R_none, gas_component):
            return rho_gas_component(R_none*u.kpc, gas_component) / cst.m_p.si


        def rho_gas(R_none):
            return rho_gas_component(R_none,"cold") + rho_gas_component(R_none,"warm") + rho_gas_component(R_none,"warm-hot") + rho_gas_component(R_none,"hot")

        def n_vm(R_none):
            return rho_gas(R_none) / cst.m_p.si

        ionized_gas = True
    ########################################################################

    if disp:
        plt.figure(dpi=100,figsize=(3,3))
        plt.plot(np.arange(l_min,l_max,0.1), n_vm(np.arange(l_min,l_max,0.1)))
        plt.xlabel("$R$ [kpc]")
        plt.ylabel("$n_{VM}$ [cm$^{-3}$]")
        #plt.yscale("log")
        plt.title(vm_model, size=15)
        #plt.savefig(dm_model_name+".png", bbox_inches='tight')
        plt.show()

    ########################################################################



    # some resolution parameters
    print("NSIDE is", NSIDE)
    #NSIDE = 2**6
    wavelength_band_resolution = 30
    #frequency_band_resolution = 0.5e14

    frequency_band_resolution = 1e9


    NPIX = hp.nside2npix(NSIDE)
    print("NPIX is", NPIX)
    dOmega = hp.nside2pixarea(nside=NSIDE)
    print("dOmega is", dOmega)
    remove_low_lat = False

    # array of theta, phi, for each pixel
    theta, phi = hp.pix2ang(nside = NSIDE, ipix = list(range(NPIX)))
    low_lat_filter = (np.degrees(theta) > 73) & (np.degrees(theta) < 113)
    high_lat_filter = (np.degrees(theta) < 73) | (np.degrees(theta) > 113)

    # frequency band
    frequency_band = np.arange(f_min_hz.value, f_max_hz.value, frequency_band_resolution) * u.Hz

    #print("using NSIDE of:", NSIDE)
    #print("     dOmega is:", dOmega)





    # sightline elements
    l_min_kpc = l_min*u.kpc
    l_max_kpc = l_max*u.kpc
    l_list = np.arange(l_min_kpc.value,l_max_kpc.value,dl.value)[:, np.newaxis]

    # 2D array of distance elements along each sightline, sun-centric
    l_array = np.ones((len(l_list),len(theta))) * l_list
    if remove_low_lat:             # number of pixels
        l_array[:,low_lat_filter] = 0

    # 2D array of distance elements along each sightline, galaxy-centric
    x = l_array * np.sin(theta) * np.cos(phi) - sun_mw_distance_kpc.value
    y = l_array * np.sin(theta) * np.sin(phi)
    z = l_array * np.cos(theta)
    R_array = np.sqrt(x**2 + y**2 + z**2)

    # calculate AQN number density
    n_aqn_cm3 = n_halo(R_array)

    h_func_cutoff = 17 + 12*np.log(2)
    def h(x):
        return_array = np.copy(x)
        return_array[np.where(x<1)] = (17 - 12*np.log(x[np.where(x<1)]/2))
        return_array[np.where(x>=1)] = h_func_cutoff
        return return_array

    if not ionized_gas:


        # calculate VM number density
        n_vm_cm3 = n_vm(R_array) 

        # calculate AQN temperature
        def T_AQN_neutral(n_bar, Dv, f, g):
            return (3/32 * np.pi * (1-g) * f * Dv * 1/(cst.alpha**(5/2)) * m_p_erg * (m_e_erg)**(1/4) * 
                    ((n_bar).to(1/u.cm**3) * invcm_to_erg**3))**(4/17) * erg_to_eV



        # erg Hz^-1 s^-1 cm^-2
        # nu Hz
        # T eV
        def spectral_surface_emissivity_cgs(nu, T):
            T_K = T * eV_to_K
            k_B_cgs = cst.k_B.cgs
            h_cgs = cst.h.cgs

            return  (1/u.Hz)*(1/cst.hbar.cgs * 1/cst.c.cgs)**2 * 4/45 * (k_B_cgs * T_K)**3 * cst.alpha ** (5/2) * 1/np.pi * ((k_B_cgs * T_K)/(m_e_eV*eV_to_erg))**(1/4) * (1 + (h_cgs * nu)/(k_B_cgs * T_K)) * np.exp(- (h_cgs * nu)/(k_B_cgs * T_K)) * h((h_cgs * nu)/(k_B_cgs * T_K))/u.s


        T_aqn_ev = T_AQN_neutral(n_vm_cm3, Dv, f, g)

        if disp:
            plt.figure(dpi=100,figsize=(3,3))
            plt.plot(l_list,T_aqn_ev[:,500], ".")
            plt.xlabel("$L$ [kpc]")
            plt.ylabel("$T_AQN$ [eV]")
            plt.show()


        # integrate along the bandwidth
        # calculate spectral spatial emissivity
        dFdw_erg_hz_cm2 = np.zeros(np.shape(T_aqn_ev)) * skymap_units #u.erg/u.cm**2/u.s/u.Hz
        for nu in frequency_band:
            #print(nu)
            dFdw_erg_hz_cm2 += convert_to_skymap_units(spectral_surface_emissivity_cgs(nu, T_aqn_ev), nu)

        if disp:
            plt.figure(dpi=100,figsize=(3,3))
            plt.plot(l_list,dFdw_erg_hz_cm2[:,500], ".")
            plt.xlabel("$L$ [kpc]")
            plt.ylabel("Surface Emissiv. ["+str(dFdw_erg_hz_cm2.unit)+"]")
            plt.show()


        # calculate spectral spatial emissivity
        dFdomega_erg_hz_cm3 = dFdw_erg_hz_cm2  * R_aqn_cm**2 * n_aqn_cm3.to(1/u.cm**3) #*4*np.pi

        if disp:
            plt.figure(dpi=100,figsize=(3,3))
            plt.plot(l_list,dFdomega_erg_hz_cm3[:,500], ".")
            plt.xlabel("$L$ [kpc]")
            plt.ylabel("Spatial Emissiv. ["+str(dFdomega_erg_hz_cm3.unit)+"]")
            plt.show()

        # integrate along sightline
        F_tot_erg_hz_cm2 = np.sum(dFdomega_erg_hz_cm3,0) /(4*np.pi*u.sr) * dl.cgs * dOmega ##############
        if remove_low_lat:
            F_tot_erg_hz_cm2[low_lat_filter] = 0
        #F_tot_erg_hz_cm2[low_lat_filter] = 0

    else:
        # calculate mass densities of each gas component
        n_cold     = n_gas_component(R_array, "cold") 

        n_warm     = n_gas_component(R_array, "warm") # avoid setting variables
        T_warm     = (3e4 + 1e5)/2 * u.K * K_to_eV

        n_warm_hot = n_gas_component(R_array, "warm-hot")
        T_warm_hot = (1e5 + 1e6)/2 * u.K * K_to_eV

        n_hot      = n_gas_component(R_array, "hot")
        T_hot      = 1e6 * u.K * K_to_eV



        spectral_surface_emissivity_u_factor = (1 / cst.hbar.cgs) * (1/(cst.hbar.cgs * cst.c.cgs))**2 * (cst.hbar.cgs * 1/u.Hz * 1/u.s)

        def spectral_surface_emissivity(T, frequency_band):
            res = np.zeros(np.shape(T)) * u.mK #* skymap_units / u.sr # u.photon / u.cm**2 / u.s / u.Angstrom / u.sr  # * u.erg/u.Hz/u.s/u.cm**2/u.sr# # 
            res[T<0] = -1 * u.mK #* skymap_units / u.sr # u.photon / u.cm**2 / u.s / u.Angstrom / u.sr  # * u.erg/u.Hz/u.s/u.cm**2/u.sr 
            f = frequency_band[0]
            T = T * eV_to_erg
            for i in range(len(frequency_band)):
                w_div_T = (2*np.pi*Hz_to_erg)*frequency_band[i] / T
                res_new = spectral_surface_emissivity_u_factor * 4/45 * \
                            T[T>0]**3 * cst.alpha ** (5/2) * 1/np.pi * \
                           (T[T>0]/(m_e_eV*eV_to_erg))**(1/4) * \
                           (1 + w_div_T[T>0]) * np.exp(- w_div_T[T>0]) * h(w_div_T[T>0]) * 1/(dOmega*u.sr)
                res_new = convert_to_skymap_units(res_new, frequency_band[i]).to(u.mK, equivalencies = u.brightness_temperature(frequency_band[i]))
                res[T>0] += res_new #* conversion_array[i] 
            return res 


        def spectral_spatial_emissivity_cold(n_AQN, n_bar, Dv, f, g, frequency_band):
            T_AQN = T_AQN_analytical(n_cold, Dv, f, g)
            dFdw = spectral_surface_emissivity(T_AQN, frequency_band)
            return dFdw 

        def spectral_spatial_emissivity_hot(n_AQN, n_bar, Dv, f, g, T_gas, R_AQN, nu):
            T_neu_eV = T_AQN_analytical(n_bar, Dv, f, g)
            T_ion_eV = T_AQN_ionized2(n_bar, Dv, f, g, T_gas, R_AQN)
            T_AQN = T_neu_eV.copy()                   # don't need this, use T_neu_eV
            ion_greater_neu = np.where(T_ion_eV > T_neu_eV)
            T_AQN[ion_greater_neu] = T_ion_eV[ion_greater_neu]
            dFdw = spectral_surface_emissivity(T_AQN, frequency_band)
            #dF = np.sum(dFdw,2)
            return dFdw

        def calc_T_aqn(n_AQN, n_bar, Dv, f, g, T_gas, R_AQN, nu):
            T_neu_eV = T_AQN_analytical(n_bar, Dv, f, g)
            T_ion_eV = T_AQN_ionized2(n_bar, Dv, f, g, T_gas, R_AQN)
            T_AQN = T_neu_eV.copy()                   # don't need this, use T_neu_eV
            ion_greater_neu = np.where(T_ion_eV > T_neu_eV)
            T_AQN[ion_greater_neu] = T_ion_eV[ion_greater_neu]
            dFdw = spectral_surface_emissivity(T_AQN, frequency_band)
            #dF = np.sum(dFdw,2)
            return T_AQN

        def func(n_c, n_w, n_wh,n_h, frequency_band):
            #print(spectral_spatial_emissivity_cold(n_aqn_cm3, n_c,  Dv, f, g,                    frequency_band).unit)
            epsilon = (spectral_spatial_emissivity_cold(n_aqn_cm3, n_c,  Dv, f, g,                    frequency_band) +
                       spectral_spatial_emissivity_hot( n_aqn_cm3, n_w,  Dv, f, g, T_warm,     R_aqn_cm, frequency_band) +
                       spectral_spatial_emissivity_hot( n_aqn_cm3, n_wh, Dv, f, g, T_warm_hot, R_aqn_cm, frequency_band) +
                       spectral_spatial_emissivity_hot( n_aqn_cm3, n_h,  Dv, f, g, T_hot,      R_aqn_cm, frequency_band))* \
                4 * np.pi * R_aqn_cm**2 * n_aqn_cm3.to(1/u.cm**3)
            print(epsilon)
            return epsilon #.value

        F_tot_galex = sum(func(n_cold,n_warm,n_warm_hot,n_hot, frequency_band)) / (4*np.pi) * (dl.cgs) * dOmega

        T_aqn_cold_ev     = T_AQN_analytical(n_cold, Dv, f, g)
        T_aqn_warm_ev     = calc_T_aqn( n_aqn_cm3, n_warm,  Dv, f, g, T_warm,     R_aqn_cm, frequency_band)
        T_aqn_warm_hot_ev = calc_T_aqn( n_aqn_cm3, n_warm_hot, Dv, f, g, T_warm_hot, R_aqn_cm, frequency_band)
        T_aqn_hot_ev      = calc_T_aqn( n_aqn_cm3, n_hot,  Dv, f, g, T_hot,      R_aqn_cm, frequency_band)

        T_aqn_dict_ev = {"cold":     T_aqn_cold_ev, 
                         "warm":     T_aqn_warm_ev,
                         "warm_hot": T_aqn_warm_hot_ev,
                         "hot":      T_aqn_hot_ev}

        spectral_spatial_emissivity = func(n_cold,n_warm,n_warm_hot,n_hot, frequency_band)
        print(np.min(spectral_spatial_emissivity), np.max(spectral_spatial_emissivity))
        #print(np.shape(func(n_cold,n_warm,n_warm_hot,n_hot, frequency_band)))
    #F_tot_erg_hz_cm2 = calc_F_tot("NFW Profile", "Constant VM density", 0.6,50, dl = 0.5 * u.kpc, NSIDE = 2**6)
    #F_tot_erg_hz_cm2 = calc_F_tot("NFW Profile", "Gas Components", 0.6,50, dl = 0.5 * u.kpc)
    #create_skymap(F_tot_erg_hz_cm2)
    print("done")

    return F_tot_galex
