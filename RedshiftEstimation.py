from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import pandas as pd
import re

# Galaxies given by Connor
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00002332_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00002332_clear-prism_v1.0_x1d.fits"
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00006438_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00006438_clear-prism_v1.0_x1d.fits"
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00008113_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00008113_clear-prism_v1.0_x1d.fits"
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00017566_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00017566_clear-prism_v1.0_x1d.fits"

# Comparison with Bunker et al. 2024
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-10058975_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-10058975_clear-prism_v1.0_x1d.fits" # z = 9.433
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00021842_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00021842_clear-prism_v1.0_x1d.fits" # z = 7.981
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00018846_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00018846_clear-prism_v1.0_x1d.fits" # z = 6.336
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00022251_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00022251_clear-prism_v1.0_x1d.fits" # z = 5.800
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0_x1d.fits" # z = 4.776
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00003892_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00003892_clear-prism_v1.0_x1d.fits" # z = 2.227

# Spectra used in testing 
#fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumhst-00000755_clear-prism_v1.0_x1d.fits" # z = 3.476
#fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumhst-00000930_clear-prism_v1.0_x1d.fits" # z = 4.062
#fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumhst-00000998_clear-prism_v1.0_x1d.fits" # z = 4.422
#fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumjwst-00045170_clear-prism_v1.0_x1d.fits" # z = 8.368
#fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumjwst-00055757_clear-prism_v1.0_x1d.fits" # z = 9.746
#fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumhst-00002389_clear-prism_v1.0_x1d.fits" # z = 7.043

fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumhst-00000604_clear-prism_v1.0_x1d.fits"

observationsFolder = "Testing_spectra/1D"
input_excel = "Data/Extracted_observations.xlsx"
output_excel = "Data/Redshift_comparison_v2.xlsx"

df = pd.read_excel(input_excel, dtype={'NIRSpec_ID': str})

# Open the FITS file and read data
with fits.open(fitsFile) as spec:
    #spec.info()
    data = spec[1].data
    wavelength = data['wavelength']
    flux = data['flux']
    fields = re.search(r"hlsp_jades_jwst_nirspec_(.+)-(\d+)_clear-prism", fitsFile)
    if fields:
        tier = fields.group(1) 
        obs_id = fields.group(2)
        
        match_row = df[(df["NIRSpec_ID"] == obs_id) & (df["TIER"] == tier)]    
        
        if not match_row.empty:
            z_spec = match_row["z_Spec"].values[0]  # Extract the redshift
            z_spec_flag = match_row["z_Spec_flag"].values[0]  # Extract the spec flag
            
            if z_spec > 0:

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

                # Normalise flux to range in values 0 to 1
                flux_min = np.min(flux)
                flux_max = np.max(flux)
                normalized_flux = (flux - flux_min) / (flux_max - flux_min)

                # Define rest-frame wavelengths of key emission lines (in Å)
                emission_lines = {
                    'OII_3727': 3727,
                    'H_beta' : 4861,
                    'OIII_4959': 4959,
                    'OIII_5007': 5007,
                    'H_alpha': 6563,
                    'NII_6583': 6583,
                    'SIII': 9531
                }

                # Redshifts that will be checked 
                z_min, z_max, z_step = 2, 15, 0.001
                redshifts = np.arange(z_min, z_max, z_step)

                best_redshift = 0
                best_score = 0
                final_peak_count = 0
                tolerance = 100

                # Parameters for peak detection
                threshold = 0.2  # Minimum height for normalized flux
                prominence = 0.04  # Minimum prominence for peaks

                # Detect peaks in the normalized flux (used for estimation)
                peaks, properties = find_peaks(
                    normalized_flux, 
                    height=threshold, 
                    prominence=prominence, 
                )

                # peak_wavelengths = wavelength_angstrom[peaks]
                # peak_flux_values = normalized_flux[peaks]

                # print("Wavelengths at peaks:", peak_wavelengths)
                # print("Flux values at peaks:", peak_flux_values)

                #Fit a high-order polynomial and normalise the flux
                p = Polynomial.fit(wavelength_angstrom, flux, deg=7)
                baseline = p(wavelength_angstrom)
                flux_baseline_corrected = flux - baseline
                flux_min = np.min(flux_baseline_corrected)
                flux_max = np.max(flux_baseline_corrected)
                normalized_corrected_flux = (flux_baseline_corrected - flux_min) / (flux_max - flux_min)

                # Interpolator, used to find flux at observer wavelegths
                flux_interpolator = interp1d(wavelength_angstrom, normalized_corrected_flux, bounds_error=False, fill_value=0)
                flux_interpolator_snr = interp1d(wavelength_angstrom, normalized_flux, bounds_error=False, fill_value=0)

                
                snr_window_size = 5000

                for z in redshifts:
                    score = 0
                    matching_peak_count = 0

                    # Contributions dictionaries for each redshift

                    for line_name, rest_wavelength in emission_lines.items():
                        # Calculate observed wavelength
                        observed_wavelength = rest_wavelength * (1 + z)
                        
                        if observed_wavelength <= np.max(wavelength_angstrom) and observed_wavelength >= np.min(wavelength_angstrom):
                        
                            # Interpolate the flux at the observed wavelength
                            observed_flux = flux_interpolator(observed_wavelength)
                            
                            #weight = np.log1p(observed_wavelength / cutoff_wavelength) / 2 + 0.5
                            
                            # Check for peaks near the observed wavelength
                            for peak in peaks:
                                #Calculate the distance to the peak from observed wavelength
                                distance = np.abs(wavelength_angstrom[peak] - observed_wavelength)
                                if  distance < tolerance:
                                    #Calculate the window for snr
                                    window_indices = (np.abs(wavelength_angstrom - wavelength_angstrom[peak]) < snr_window_size)
                                    #Estimate local noise level
                                    local_noise = np.std(normalized_flux[window_indices])
                                    original_observed_flux = flux_interpolator_snr(observed_wavelength)
                                    snr = original_observed_flux / (local_noise + 1e-6)
                                    snr_weight = np.clip(snr / 10, 0, 1)
                                    
                                    # Calculate the score using the flux at the observed wavelength
                                    score += observed_flux * snr_weight
                                    #matching_peak_count += 1
                                    

                    # Make sure that a lot of small peaks close to the observed wavelenghts don't skew the result
                    # if matching_peak_count > 0:
                    #     score /= matching_peak_count
                        

                    # Update best redshift if score is higher
                    if score > best_score:
                        best_score = score
                        best_redshift = z
                        #final_peak_count = matching_peak_count

                if best_redshift != 0:
                    print(f"Score: {best_score}")
                    print(f"Best Redshift: {best_redshift}")
                    # print(f"Matching peaks: {final_peak_count}")
                    # for line_name, contribution in best_line_contributions.items():
                    #     print(f"{line_name}: {contribution}")

                    # Plot the spectrum with identified peaks
                    plt.figure(figsize=(10, 6))
                    plt.plot(wavelength_angstrom, flux, lw=1, color='black', label="Observed Flux")
                    plt.plot(wavelength_angstrom, flux_baseline_corrected, lw=1, color='grey', alpha=0.5, label="Flux with Baseline (Polynomial Fit)")
                    plt.scatter(wavelength_angstrom[peaks], flux[peaks], color='blue', label="Detected Peaks")
                    plt.xlabel('Wavelength (Å)', fontsize=14)
                    plt.ylabel('Flux (erg/s/cm$^{2}$/Å)', fontsize=14)
                    plt.title('1D Spectrum', fontsize=16)

                    # Make the grid higher, to show the emssion lines above the plot
                    current_ylim = plt.ylim()
                    plt.ylim(current_ylim[0], max(flux) * 1.2)  

                    # Display redshifted emission lines with labels
                    for line_name, rest_wavelength in emission_lines.items():
                        observed_wavelength = rest_wavelength * (1 + best_redshift)
                        
                        # Plot the emission lines according to catalogue redshift
                        observed_wavelength_zspec = rest_wavelength * (1 + z_spec)
                        
                        if observed_wavelength <= np.max(wavelength_angstrom):
                            plt.axvline(observed_wavelength, color='red', linestyle='--', alpha=0.8, label="_nolegend_")
                            plt.text(observed_wavelength + 50, max(flux) * 1.05, line_name, 
                                    rotation=90, verticalalignment='bottom', fontsize=8, color='black')
                            
                            if (abs(best_redshift - z_spec) > 0.2):
                                plt.axvline(observed_wavelength_zspec, color='purple', linestyle='--', alpha=0.8, label="_nolegend_")
                                plt.text(observed_wavelength_zspec + 50, max(flux) * 1.1, line_name + " (Spec)", 
                                        rotation=90, verticalalignment='bottom', fontsize=8, color='blue')

                            
                            # Display rest-frame lines
                            plt.axvline(rest_wavelength, color='green', linestyle='--', alpha=0.8, label="_nolegend_")
                            
                    #Extend the grid on the left to make sure all of the rest-frame lines are visible
                    current_xlim = plt.xlim()
                    plt.xlim(current_xlim[0] - (current_xlim[1] - current_xlim[0]) * 0.05, current_xlim[1])

                    # Add custom legend entries for redshifted and rest-frame lines
                    if (abs(best_redshift - z_spec) > 0.2):
                        custom_legend = [
                            Line2D([0], [0], color='red', linestyle='--', label=f'Redshifted Lines (z={best_redshift:.3f})'),
                            Line2D([0], [0], color='purple', linestyle='--', label=f'Redshifted Lines (z_spec={z_spec:.3f})'),
                            Line2D([0], [0], color='green', linestyle='--', label='Rest-Frame Emission Lines'),
                        ]
                    else:
                        custom_legend = [
                            Line2D([0], [0], color='red', linestyle='--', label=f'Redshifted Lines (z={best_redshift:.3f})'),
                            Line2D([0], [0], color='green', linestyle='--', label='Rest-Frame Emission Lines'),
                        ]
                    
                    plt.legend(handles=custom_legend + plt.gca().get_legend_handles_labels()[0],
                            title=f'Redshift: {best_redshift:.3f}', title_fontproperties={'weight': 'bold'})
                    
                    plt.grid(True)
                    plt.show()
                else:
                    print("Redshift could not be detected")