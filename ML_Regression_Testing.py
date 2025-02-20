import os
import re
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import tensorflow as tf


model = tf.keras.models.load_model("redshift_estimator_model.keras", compile=False)

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
    # Parse the filename to extract tier and observation ID
    fields = re.search(r"hlsp_jades_jwst_nirspec_(.+)-(\d+)_clear-prism", fits_file)
    if fields:
        tier = fields.group(1)
        obs_id = fields.group(2)
        
        # Look up the corresponding row in the catalog
        match_row = df_catalog[(df_catalog["NIRSpec_ID"] == obs_id) & (df_catalog["TIER"] == tier)]
        if not match_row.empty:
            # Only use observations with z_Spec_flag 'A' (if needed)
            if match_row["z_Spec_flag"].values[0] != 'A':
                continue

            catalog_z = match_row["z_Spec"].values[0]
            
            # Full path to the FITS file
            fits_path = os.path.join(observationsFolder, fits_file)
            with fits.open(fits_path) as spec:
                data = spec[1].data
                # Convert wavelength to Angstrom
                wavelength = data['wavelength'] * 1e4
                flux = data['flux']
            
            # Remove NaN values
            valid = ~np.isnan(flux)
            wavelength = wavelength[valid]
            flux = flux[valid]
            
            # Ensure the arrays are sorted by wavelength
            sort_idx = np.argsort(wavelength)
            wavelength = wavelength[sort_idx]
            flux = flux[sort_idx]
            
            # Normalize the flux using min-max normalization
            flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
            
            # Interpolate the normalized flux onto the common wavelength grid
            interp_func = interp1d(wavelength, flux_norm, bounds_error=False, fill_value=0)
            flux_resampled = interp_func(wavelength_grid)
            
            # Reshape to match the input shape expected by the model: (1, n_bins, 1)
            flux_input = flux_resampled.reshape(1, n_bins, 1)
            
            # Predict the redshift using the model
            pred = model.predict(flux_input)
            predicted_redshift = pred[0, 0]
            
            # Save the result
            results.append({
                "Filename": fits_file,
                "Predicted Redshift": predicted_redshift,
                "Catalog Redshift": catalog_z
            })

df_results = pd.DataFrame(results)
output_excel = "Data/Redshift_predictions.xlsx"
df_results.to_excel(output_excel, index=False)
print("Results saved to", output_excel)
