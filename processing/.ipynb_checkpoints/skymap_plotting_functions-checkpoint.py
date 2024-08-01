import matplotlib.pyplot as plt

import numpy as np

from astropy import constants as cst
from astropy import units as u
import healpy as hp


def create_skymap2(signal, signal_unit, log_signal):
    f1 = plt.figure(figsize=(7.5,4.5), dpi=700)
    # rotation in (lat, long, psi)
    if log_signal:
        hp.mollview(np.log10(signal.value), 
                    title="", 
                    rot=(0,0,0),
                    cmap='gist_rainbow', 
                    fig=f1, 
                    unit="")
    else:
        hp.mollview(signal.value, 
                    title="", 
                    rot=(0,0,0),
                    cmap='gist_rainbow', 
                    fig=f1, 
                    unit="")    
    #hp.graticule(dmer=360,dpar=360,alpha=0)  
    hp.graticule(dmer=30,dpar=30, alpha=1, color="white", linewidth=10)
    longitude_labels = [150,120,90,60,30,0,330,300,270,240,210]
    for i in range(len(longitude_labels)):
        plt.text((-5+i)*0.34,0.1,str(longitude_labels[i])+"$^\circ$",size=7,horizontalalignment="center")

    latitude_labels = [-60,-30,0,30,60]
    plt.text(-2.01,-0.41,"-30$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-2.1,0,"0$^\circ$",size=7,horizontalalignment="center")
    plt.text(-2,0.39,"30$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-1.5,0.73,"60$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-1.51,-0.8,"-60$^\circ$",size=7,horizontalalignment="center") 
    if log_signal:
        plt.text(0,-1.10,"Logarithmic",size=9,horizontalalignment="center")
    #plt.text(0,-1.37,"photons$\,\cdot\,$cm$^{-2}\cdot$A$^{-1}\cdot$s$^{-1}\cdot$sr$^{-1}$",size=10,horizontalalignment="center")
    plt.text(0,-1.37,signal_unit,size=10,horizontalalignment="center")
    plt.savefig("simulated_signal_sky_map.png")
    plt.show()

    

def create_skymap(signal):
    f1 = plt.figure(figsize=(7.5,4.5), dpi=700)
    # rotation in (lat, long, psi)
    hp.mollview(np.log10(signal.value), 
                title="FUV Emissions from AQN Annihilation", 
                rot=(0,0,0),
                cmap='gist_rainbow', 
                fig=f1, 
                unit="")

    #hp.graticule(dmer=360,dpar=360,alpha=0)  
    hp.graticule(dmer=30,dpar=30, alpha=1, color="white", linewidth=10)
    longitude_labels = [150,120,90,60,30,0,330,300,270,240,210]
    for i in range(len(longitude_labels)):
        plt.text((-5+i)*0.34,0.1,str(longitude_labels[i])+"$^\circ$",size=7,horizontalalignment="center")

    latitude_labels = [-60,-30,0,30,60]
    plt.text(-2.01,-0.41,"-30$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-2.1,0,"0$^\circ$",size=7,horizontalalignment="center")
    plt.text(-2,0.39,"30$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-1.5,0.73,"60$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-1.51,-0.8,"-60$^\circ$",size=7,horizontalalignment="center") 

    plt.text(0,-1.10,"Logarithmic",size=9,horizontalalignment="center")
    #plt.text(0,-1.37,"photons$\,\cdot\,$cm$^{-2}\cdot$A$^{-1}\cdot$s$^{-1}\cdot$sr$^{-1}$",size=10,horizontalalignment="center")
    plt.text(0,-1.37,"Jy/sr",size=10,horizontalalignment="center")
    #plt.savefig("BURKERT & CONST VM 5eV.png")
    plt.show()

def create_skymap_T_aqn(signal, disp_text):
    f1 = plt.figure(figsize=(7.5,4.5), dpi=700)
    # rotation in (lat, long, psi)
    hp.mollview(signal.value, 
                title="$T_{AQN}$, "+str(disp_text), 
                rot=(0,0,0),
                cmap='gist_rainbow', 
                fig=f1, 
                unit="")

    #hp.graticule(dmer=360,dpar=360,alpha=0)  
    hp.graticule(dmer=30,dpar=30, alpha=1, color="white", linewidth=10)
    longitude_labels = [150,120,90,60,30,0,330,300,270,240,210]
    for i in range(len(longitude_labels)):
        plt.text((-5+i)*0.34,0.1,str(longitude_labels[i])+"$^\circ$",size=7,horizontalalignment="center")

    latitude_labels = [-60,-30,0,30,60]
    plt.text(-2.01,-0.41,"-30$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-2.1,0,"0$^\circ$",size=7,horizontalalignment="center")
    plt.text(-2,0.39,"30$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-1.5,0.73,"60$^\circ$",size=7,horizontalalignment="center") 
    plt.text(-1.51,-0.8,"-60$^\circ$",size=7,horizontalalignment="center") 

    #plt.text(0,-1.10,"",size=9,horizontalalignment="center")
    plt.text(0,-1.37,"eV",size=10,horizontalalignment="center")

    #plt.savefig("BURKERT & CONST VM 5eV.png")
    plt.show()
