import os
import re
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import tensorflow as tf
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import find_peaks

# Load the model saved in the TF-Keras native format.
model = tf.keras.models.load_model("redshift_estimator_model.keras", compile=False)

# These values should match those used during training.
common_min = 6000   
common_max = 56000  
n_bins = 2048      
wavelength_grid = np.linspace(common_min, common_max, n_bins)


input_excel = "Data/Extracted_observations.xlsx"
df_catalog = pd.read_excel(input_excel, dtype={'NIRSpec_ID': str})


observationsFolder = "Testing_spectra/1D"
results = []
fits_files = sorted(os.listdir(observationsFolder))
fits_files_test = fits_files[-1000:]

for fits_file in fits_files_test:
    # Parse filename to extract tier and observation ID
    fields = re.search(r"hlsp_jades_jwst_nirspec_(.+)-(\d+)_clear-prism", fits_file)
    if not fields:
        continue
    tier = fields.group(1)
    obs_id = fields.group(2)
    
    # Look up the corresponding catalog row
    match_row = df_catalog[(df_catalog["NIRSpec_ID"] == obs_id) & (df_catalog["TIER"] == tier)]
    if match_row.empty:
        continue
    # Only use observations with z_Spec_flag 'A'
    if match_row["z_Spec_flag"].values[0] != 'A':
        continue
    catalog_z = match_row["z_Spec"].values[0]
    if catalog_z < 2:
        continue
    
    # Full path to the FITS file
    fits_path = os.path.join(observationsFolder, fits_file)
    with fits.open(fits_path) as spec:
        data = spec[1].data
        wavelength = data['wavelength']
        flux = data['flux']
        
    # Remove NaN values and sort by wavelength
    valid = ~np.isnan(flux)
    wavelength = wavelength[valid]
    flux = flux[valid]
    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    flux = flux[sort_idx]
    
    # Convert wavelength to Angstrom
    wavelength_angstrom = wavelength * 1e4
    
    # Normalize flux using min-max normalization
    flux_min = np.min(flux)
    flux_max = np.max(flux)
    normalized_flux = (flux - flux_min) / (flux_max - flux_min)
    
    # Interpolate normalized flux onto the common wavelength grid
    interp_func = interp1d(wavelength_angstrom, normalized_flux, bounds_error=False, fill_value=0)
    flux_resampled = interp_func(wavelength_grid)
    
    # Prepare input for the regression model
    flux_input = flux_resampled.reshape(1, n_bins, 1)
    
    pred = model.predict(flux_input)
    predicted_redshift = pred[0, 0]
    
    # Define rest-frame wavelengths of key emission lines (in Å)
    emission_lines = {
        'OII_3727': 3727,
        'H_beta' : 4861,
        'OIII_4959': 4959,
        'OIII_5007': 5007,
        'H_alpha': 6563,
        'SIII': 9531
    }

    # Redshifts that will be checked 
    search_z_min = max(2, predicted_redshift - 3)
    search_z_max = predicted_redshift + 3
    z_step = 0.0001
    redshifts = np.arange(search_z_min, search_z_max, z_step)

    best_redshift = 0
    best_score = 0
    final_peak_count = 0
    tolerance = 20

    # Parameters for peak detection
    threshold = 0.1 
    prominence = 0.03  

    # Detect peaks in the normalized flux 
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

    # cutoff_wavelength = 8000
    snr_window_size = 5000

    for z in redshifts:
        score = 0
        matching_peak_count = 0

        # Contributions dictionaries for each redshift
        current_line_contributions = {line_name: 0 for line_name in emission_lines.keys()} 
        current_line_peak_counts = {line_name: 0 for line_name in emission_lines.keys()}

        for line_name, rest_wavelength in emission_lines.items():
            # Calculate observed wavelength
            observed_wavelength = rest_wavelength * (1 + z)
            
            if observed_wavelength <= np.max(wavelength_angstrom) and observed_wavelength >= np.min(wavelength_angstrom):
            
                # Interpolate the flux at the observed wavelength
                observed_flux = flux_interpolator(observed_wavelength)
                
                #Calculate the weight for the emission line
                #weight = np.sqrt(observed_wavelength / cutoff_wavelength)
                #weight = np.log1p(observed_wavelength / cutoff_wavelength) / 2 + 0.5
                #weight = (np.log1p(observed_wavelength) / np.sqrt(cutoff_wavelength)) + 1

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
                        
                        break

        # Make sure that a lot of small peaks close to the observed wavelenghts don't skew the result
        # if matching_peak_count > 0:
        #     score /= matching_peak_count

        # Update best redshift if score is higher
        if score > best_score:
            best_score = score
            best_redshift = z
            #final_peak_count = matching_peak_count

    results.append({
        "Filename": fits_file,
        "Regression Prediction": predicted_redshift,
        "Detailed Redshift": best_redshift,
        "Catalog Redshift": catalog_z
    })
    
df_results = pd.DataFrame(results)
output_excel = "Data/Hybrid_model.xlsx"
df_results.to_excel(output_excel, index=False)
print("Results saved to", output_excel)