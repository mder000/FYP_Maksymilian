from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Galaxies given by Connor
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00002332_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00002332_clear-prism_v1.0_x1d.fits"
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00006438_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00006438_clear-prism_v1.0_x1d.fits"
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00008113_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00008113_clear-prism_v1.0_x1d.fits"
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00017566_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00017566_clear-prism_v1.0_x1d.fits"

# Comparison with Bunker et al. 2024
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-10058975_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-10058975_clear-prism_v1.0_x1d.fits" # z = 9.433 (matching peaks causes an error here)
fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00021842_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00021842_clear-prism_v1.0_x1d.fits" # z = 7.981
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00018846_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00018846_clear-prism_v1.0_x1d.fits" # z = 6.336 (big spike at the beggining causes an error)
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00022251_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00022251_clear-prism_v1.0_x1d.fits" # z = 5.800
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0_x1d.fits" # z = 4.776

# Open the FITS file and read data
with fits.open(fitsFile) as spec:
    #spec.info()
    data = spec[1].data
    wavelength = data['wavelength']
    flux = data['flux']

# print(f"Length of wavelength: {len(wavelength)}")
# print(f"Length of flux: {len(flux)}")

# Convert wavelength to Angstrom
wavelength_angstrom = wavelength * 1e4

# Delete Nan values
valid_indices = ~np.isnan(flux)  
flux = flux[valid_indices]     
wavelength_angstrom = wavelength_angstrom[valid_indices] 

# print(f"Length of wavelength: {len(wavelength_angstrom)}")
# print(f"Length of flux: {len(flux)}")

# Normalise flux, so that very big peaks don't skew the result
flux_min = np.min(flux)
flux_max = np.max(flux)
normalized_flux = (flux - flux_min) / (flux_max - flux_min)

# Define rest-frame wavelengths of key emission lines (in Å)
emission_lines = {
    'H_beta' : 4861,
    'OIII_4959': 4959,
    'OIII_5007': 5007,
    'H_alpha': 6563,
    #'Lyman_alpha': 1216,
}

# Redshifts that will be checked 
z_min, z_max, z_step = 0, 10, 0.001
redshifts = np.arange(z_min, z_max, z_step)

best_redshift = 0
best_score = 0
tolerance = 600

# Parameters for peak detection
threshold = 0.1  # Minimum height for normalized flux
prominence = 0.02  # Minimum prominence for peaks

# Detect peaks in the normalized flux (used for estimation)
peaks, properties = find_peaks(
    normalized_flux, 
    height=threshold, 
    prominence=prominence, 
)

peak_wavelengths = wavelength_angstrom[peaks]
peak_flux_values = normalized_flux[peaks]

# print("Wavelengths at peaks:", peak_wavelengths)
# print("Flux values at peaks:", peak_flux_values)

# Interpolator, used to find flux at observer wavelegths
flux_interpolator = interp1d(wavelength_angstrom, normalized_flux, bounds_error=False, fill_value=0)

# Dictionary to track how each emition line contributes to the final score
best_line_contributions = {line_name: 0 for line_name in emission_lines.keys()}

for z in redshifts:
    score = 0
    matching_peak_count = 0

    # Contributions dictionaries for each redshift
    current_line_contributions = {line_name: 0 for line_name in emission_lines.keys()} 
    current_line_peak_counts = {line_name: 0 for line_name in emission_lines.keys()}

    for line_name, rest_wavelength in emission_lines.items():
        # Calculate observed wavelength
        observed_wavelength = rest_wavelength * (1 + z)
        
        # Interpolate the flux at the observed wavelength
        observed_flux = flux_interpolator(observed_wavelength)

        # Check for peaks near the observed wavelength
        for peak in peaks:
            #Calculate the distance to the peak from observed wavelength
            distance = np.abs(wavelength_angstrom[peak] - observed_wavelength)
            if  distance < tolerance:
                # Calculate the score using the flux at the observed wavelength
                score += observed_flux
                matching_peak_count += 1
                
                # Tracking the contributions
                current_line_contributions[line_name] += observed_flux
                current_line_peak_counts[line_name] += 1
    
    # Make sure that a lot of small peaks close to the observed wavelenghts don't skew the result
    if matching_peak_count > 0:
        score /= matching_peak_count
        
    # Normalize each emission line's contribution for debugging
    for line_name in current_line_contributions:
        if current_line_peak_counts[line_name] > 0:
            current_line_contributions[line_name] /= current_line_peak_counts[line_name]

    # Update best redshift if score is higher
    if score > best_score:
        best_score = score
        best_redshift = z
        best_line_contributions = current_line_contributions.copy() 

if best_redshift != 0:
    print(f"Score: {best_score}")
    print(f"Best Redshift: {best_redshift}")
    for line_name, contribution in best_line_contributions.items():
        print(f"{line_name}: {contribution}")

    # Plot the spectrum with identified peaks
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength_angstrom, flux, lw=1, color='black')
    plt.scatter(wavelength_angstrom[peaks], flux[peaks], color='red')
    plt.xlabel('Wavelength (Å)', fontsize=14)
    plt.ylabel('Flux (erg/s/cm$^{2}$/Å)', fontsize=14)
    plt.title('1D Spectrum', fontsize=16)

    # Overlay vertical lines for the best redshifted emission lines
    for line_name, rest_wavelength in emission_lines.items():
        observed_wavelength = rest_wavelength * (1 + best_redshift)
        plt.axvline(observed_wavelength, color='grey', linestyle='--', label=f'{line_name}', alpha=0.8)

    plt.legend(title=f'Redshift: {best_redshift:.3f}')
    plt.grid(True)
    plt.show()
else:
    print("Redshift could not be detected")