import numpy as np
from astropy import constants as cst
from astropy import units as u
from constants import *
from time import time as tt
def ttt(ti, s=""):
	elapsed_time = tt() - ti
	print(f"{s}: {elapsed_time:.5f} s")

def erg_hz_cm2_to_photon_units(erg_hz_cm2, wavelength):
    return (erg_hz_cm2 * 1/cst.h * 1e-7 * 1/wavelength).value * photon_units



t = tt()
lamb = 1000

C = (erg_hz_cm2).to(photon_units*u.sr, u.spectral_density(lamb*u.AA))

print(C)
ttt(t, "1")

t = tt()
lamb = 1000

C = erg_hz_cm2_to_photon_units(erg_hz_cm2, lamb)

print(C)
ttt(t, "1")


kT = 1



x = 
print(x)
# print(x.to(u.dimensionless_unscaled))
print(((2*np.pi*cst.hbar*cst.c)/(kT*u.eV*lamb*u.AA)).to(u.dimensionless_unscaled))






# temporarily moving parallel computing part here


from joblib import Parallel, delayed
import numpy as np

def process_batch(batch_quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b):
    """Process a batch of cubes."""
    results = []
    for i in range(len(batch_quant["dark_mat"])):  # Process each cube in the batch
        cube_quant = {key: val[i:i + 1] for key, val in batch_quant.items()}
        result = compute_epsilon_velocity_integral(
            cube_quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b
        )
        results.append(result)
    return np.array(results)

def split_quant(quant, n_splits):
    """Split the quant dictionary into n_splits batches."""
    split_indices = np.array_split(range(len(quant["dark_mat"])), n_splits)
    batches = []
    for indices in split_indices:
        batch = {key: val[indices] for key, val in quant.items()}
        batches.append(batch)
    return batches

def parallel_compute(quant, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b, n_jobs=-1):
    """Parallelize computation over batches of cubes."""
    # Determine the number of workers (batches) to use
    from multiprocessing import cpu_count
    n_workers = n_jobs if n_jobs > 0 else cpu_count()

    # Split quant into batches
    quant_batches = split_quant(quant, n_workers)

    # Process each batch in parallel
    batch_results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch, m_aqn_kg, band_min, band_max, adjust_T_gas, sigma_v, v_b)
        for batch in quant_batches
    )
    
    # Combine results from all batches
    # aqn_emit = np.concatenate(batch_results, axis=0)
    # aqn_emit = np.vstack(batch_results)
    print(batch_results)
    print(len(quant["dark_mat"]))
    aqn_emit = [batch_results[i][0][0] for i in range(n_workers)]
    return 

# Example usage
# aqn_emit = parallel_compute(quant, m_aqn_kg, 1300*u.AA, 1700*u.AA, False, sigma_v, v_b)
# print(aqn_emit)
# quant["aqn_emit"] = aqn_emit * epsilon_units / u.sr  # Combine back into quant
# print(quant["aqn_emit"]*(0.6*u.kpc).to(u.cm)/(4*np.pi))

