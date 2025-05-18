# Final Year Project - Redshift Estimation from 1D Spectra in the JADES Archive

## Project Overview:
This project presents an algorithm designed to estimate the redshift of galaxies based on 1D spectra from the JWST JADES (JWST Advanced Deep Extragalactic Survey) archive. The method utilizes spectra taken with the NIRSpec instrument in the clear/prism mode. The algorithm is implemented in both interactive and batch modes, and a machine learning approach has also been explored to provide an alternative estimation pathway.

## Data Preparation:
To run any of the scripts, users must first download the appropriate spectra files from the following website:

https://jades-survey.github.io/scientists/data.html

From this webpage, download "1-d and 2-d NIRSpec/MSA clear/prism spectra". The relevant filenames follow this structure:

hlsp_jades_jwst_nirspec_id-no.disperser-filter.version_extraction-type.fits

Where:

id-no = TIER + NIRSpec_ID (zero-padded to 8 digits, e.g., goods-n-mediumhst-00000604)

disperser-filter = "clear-prism"

version = version number, e.g., "v1.0"

extraction-type = "x1d" for 1D spectra (only these are used)

After downloading, place only the x1d FITS files in the following directory:

Spectra/1D/

Once the files are in place, all subsequent scripts can be executed as they rely on this folder for spectrum data and on Example_Data/Extracted_observations.xlsx for metadata.

## Available Scripts:

# RedshiftEstimationInteractive.py

- Provides an interactive redshift estimation interface.

- The user must edit the script and manually set the "fitsFile" variable at the top of the script to the path of a selected FITS file from Spectra/1D/.

# RedshiftEstimationBatch.py

- Performs redshift estimation on all spectra in Spectra/1D.

- Computes additional parameters such as SNR.

- Saves the results in the Results/ directory as an Excel spreadsheet.

# ExtractEstimatedRedshifts.py

- Generates the Example_Data/Extracted_observations.xlsx metadata file.

Must be run twice by the user:
a) Once for: Spectra/jades_dr3_prism_public_gn_v1.1.fits
b) Once for: Spectra/jades_dr3_prism_public_gs_v1.1.fits

This step is optional if the Excel file is already provided.

# RegressionEstimation.py

- Trains a machine learning model for redshift estimation using provided data.

- Outputs a trained model named "redshift_estimator_model.keras".

# redshift_estimator_model.keras

- Pre-trained model already included in the repository for inference use.

# RegressionEstimationTesting.py

- Uses the trained model to estimate redshifts on a subset of spectra from Spectra/1D.

- Results are saved to Results/Redshift_predictions.xlsx.

## Directory Structure:

Spectra/1D/ → contains downloaded 1D x1d FITS files

Example_Data/ → contains or will contain Extracted_observations.xlsx

Results/ → output folder for redshift estimation and prediction results

## Note
Ensure that all file naming follows the naming convention described above to allow the scripts to properly match spectrum files to their corresponding metadata entries.
