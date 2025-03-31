from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import os
import pandas as pd
import re

observationsFolder = "Testing_spectra/1D"
input_excel = "Data/Extracted_observations.xlsx"
output_excel = "Data/Redshift_comparison_v5.xlsx"

df = pd.read_excel(input_excel, dtype={'NIRSpec_ID': str})  # Ensure NIRSpec_ID is treated as a string
fits_files = sorted(os.listdir(observationsFolder))
results = []
obs_no = 0

for fits_file in fits_files:
    fields = re.search(r"hlsp_jades_jwst_nirspec_(.+)-(\d+)_clear-prism", fits_file)
    if fields:
        tier = fields.group(1) 
        obs_id = fields.group(2)
        
        match_row = df[(df["NIRSpec_ID"] == obs_id) & (df["TIER"] == tier)]    
        
        if not match_row.empty:
            z_spec = match_row["z_Spec"].values[0]  # Extract the redshift
            z_spec_flag = match_row["z_Spec_flag"].values[0]  # Extract the spec flag
            
            if z_spec > 2 and z_spec_flag == "A":
                fitsFile = os.path.join(observationsFolder, fits_file)
                print(f"Observation number {obs_no} started")
                
                # Open the FITS file and read data
                with fits.open(fitsFile) as spec:
                    #spec.info()
                    data = spec[1].data
                    wavelength = data['wavelength']
                    flux = data['flux']
                    flux_err = data['FLUX_ERR'] 

                # print(f"Length of wavelength: {len(wavelength)}")
                # print(f"Length of flux: {len(flux)}")
                
                #SNR calculation
                snr = flux / flux_err  
                median_snr = np.nanmedian(snr)

                # Convert wavelength to Angstrom
                wavelength_angstrom = wavelength * 1e4

                # Delete Nan values
                valid_indices = ~np.isnan(flux)  
                flux = flux[valid_indices]     
                wavelength_angstrom = wavelength_angstrom[valid_indices] 
                flux_err = flux_err[valid_indices]

                # print(f"Length of wavelength: {len(wavelength_angstrom)}")
                # print(f"Length of flux: {len(flux)}")

                # Normalise flux to range in values 0 to 1
                flux_min = np.min(flux)
                flux_max = np.max(flux)
                normalized_flux = (flux - flux_min) / (flux_max - flux_min)

                # Define rest-frame wavelengths of key emission lines (in Å)
                # emission_lines = {
                #     'OII_3727': 3727,
                #     'H_beta' : 4861,
                #     'OIII_4959': 4959,
                #     'OIII_5007': 5007,
                #     'H_alpha': 6563,
                #     'SIII': 9531
                # }
                
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
                        
                print(f"Observation number {obs_no} completed")
                obs_no += 1

                results.append({
                    "Filename": fits_file,
                    "Estimated Redshift": best_redshift,
                    "Catalog Redshift": z_spec,
                    "Catalog Spec Flag": z_spec_flag,
                    "SNR" : median_snr
                })
                
df_results = pd.DataFrame(results)
df_results.to_excel(output_excel, index=False)

print(f"Redshift comparisons saved to {output_excel}")