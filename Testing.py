from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import find_peaks, correlate
from scipy.interpolate import interp1d
import pandas as pd
import re

def resample_to_log_wavelength(wavelength, flux, num_points=5000):
    """
    Resample a spectrum onto a uniform grid in log(wavelength).
    """
    log_wave = np.log(wavelength)
    log_wave_uniform = np.linspace(log_wave.min(), log_wave.max(), num_points)
    flux_uniform = np.interp(log_wave_uniform, log_wave, flux)
    wavelength_uniform = np.exp(log_wave_uniform)
    return wavelength_uniform, flux_uniform, log_wave_uniform

def generate_template(emission_lines, wavelength_range=(3500, 10000), num_points=5000):
    template_wavelength = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
    template_flux = np.zeros_like(template_wavelength)
    for line_name, rest_wavelength in emission_lines.items():
        if wavelength_range[0] <= rest_wavelength <= wavelength_range[1]:
            sigma = 5  
            amplitude = 1.0
            template_flux += amplitude * np.exp(-0.5 * ((template_wavelength - rest_wavelength) / sigma)**2)
    return template_wavelength, template_flux


#fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumhst-00000943_clear-prism_v1.0_x1d.fits" # example file
fitsFile = "Testing_spectra/1D/hlsp_jades_jwst_nirspec_goods-n-mediumhst-00000943_clear-prism_v1.0_x1d.fits"

observationsFolder = "Testing_spectra/1D"
input_excel = "Data/Extracted_observations.xlsx"
output_excel = "Data/Redshift_comparison_v2.xlsx"

df = pd.read_excel(input_excel, dtype={'NIRSpec_ID': str})

# Open the FITS file and read data
with fits.open(fitsFile) as spec:
    data = spec[1].data
    wavelength = data['wavelength']
    flux = data['flux']
    
    fields = re.search(r"hlsp_jades_jwst_nirspec_(.+)-(\d+)_clear-prism", fitsFile)
    if fields:
        tier = fields.group(1) 
        obs_id = fields.group(2)
        
        match_row = df[(df["NIRSpec_ID"] == obs_id) & (df["TIER"] == tier)]    
        
        if not match_row.empty:
            z_spec = match_row["z_Spec"].values[0]  
            z_spec_flag = match_row["z_Spec_flag"].values[0]
            
            if z_spec > 0:
                # Convert wavelength to Angstrom
                wavelength_angstrom = wavelength * 1e4

                # Remove NaN values
                valid_indices = ~np.isnan(flux)  
                flux = flux[valid_indices]     
                wavelength_angstrom = wavelength_angstrom[valid_indices] 

                # Normalize flux (for peak detection and SNR estimation)
                flux_min = np.min(flux)
                flux_max = np.max(flux)
                normalized_flux = (flux - flux_min) / (flux_max - flux_min)


                p = Polynomial.fit(wavelength_angstrom, flux, deg=7)
                baseline = p(wavelength_angstrom)
                flux_baseline_corrected = flux - baseline
                fb_min = np.min(flux_baseline_corrected)
                fb_max = np.max(flux_baseline_corrected)
                normalized_corrected_flux = (flux_baseline_corrected - fb_min) / (fb_max - fb_min)

                # Create interpolators for use in the emission-line method
                flux_interpolator = interp1d(wavelength_angstrom, normalized_corrected_flux, bounds_error=False, fill_value=0)
                flux_interpolator_snr = interp1d(wavelength_angstrom, normalized_flux, bounds_error=False, fill_value=0)
    

                # Define rest-frame emission lines 
                emission_lines = {
                    'OII_3727': 3727,
                    'H_beta': 4861,
                    'OIII_4959': 4959,
                    'OIII_5007': 5007,
                    'H_alpha': 6563,
                    'SIII': 9531
                }
                
                # Generate a synthetic rest-frame template using the emission lines
                template_wavelength, template_flux = generate_template(emission_lines, wavelength_range=(3500, 10000), num_points=5000)
                
                # Resample both the observed (continuum-subtracted) spectrum and the template
                num_points = 5000
                obs_wave_uniform, obs_flux_uniform, log_wave_obs = resample_to_log_wavelength(wavelength_angstrom, flux_baseline_corrected, num_points)
                temp_wave_uniform, temp_flux_uniform, log_wave_temp = resample_to_log_wavelength(template_wavelength, template_flux, num_points)
                
                # Normalize both resampled spectra (mean 0, std 1)
                obs_flux_norm = (obs_flux_uniform - np.mean(obs_flux_uniform)) / np.std(obs_flux_uniform)
                temp_flux_norm = (temp_flux_uniform - np.mean(temp_flux_uniform)) / np.std(temp_flux_uniform)
                
                # Perform cross-correlation using 'same' mode
                corr = correlate(obs_flux_norm, temp_flux_norm, mode='same')
                lags = np.arange(-len(corr)//2, len(corr)//2)
                peak_index = np.argmax(corr)
                best_lag = lags[peak_index]
                
                # Determine grid spacing in log(wavelength)
                delta_log_lambda = (log_wave_obs[-1] - log_wave_obs[0]) / (num_points - 1)
                # Flip the sign of best_lag to get the correct shift
                shift_log = - best_lag * delta_log_lambda
                z_estimate_cc = np.exp(shift_log) - 1
                
                print(f"Cross-correlation estimated redshift: {z_estimate_cc:.4f}")
                
                z_min, z_max, z_step = 2, 11, 0.001
                redshifts = np.arange(z_min, z_max, z_step)
                
                best_redshift = 0
                best_score = 0
                tolerance = 100 
                snr_window_size = 5000

                # Detect peaks in the normalized flux (for emission-line matching)
                peaks, properties = find_peaks(normalized_flux, height=0.2, prominence=0.04)

                for z in redshifts:
                    score = 0
                    for line_name, rest_wavelength in emission_lines.items():
                        observed_wavelength = rest_wavelength * (1 + z)
                        if (observed_wavelength <= np.max(wavelength_angstrom)) and (observed_wavelength >= np.min(wavelength_angstrom)):
                            observed_flux = flux_interpolator(observed_wavelength)
                            for peak in peaks:
                                distance = np.abs(wavelength_angstrom[peak] - observed_wavelength)
                                if distance < tolerance:
                                    window_indices = np.abs(wavelength_angstrom - wavelength_angstrom[peak]) < snr_window_size
                                    local_noise = np.std(normalized_flux[window_indices])
                                    original_observed_flux = flux_interpolator_snr(observed_wavelength)
                                    snr = original_observed_flux / (local_noise + 1e-6)
                                    snr_weight = np.clip(snr / 10, 0, 1)
                                    score += observed_flux * snr_weight
                    if score > best_score:
                        best_score = score
                        best_redshift = z

                print(f"Emission-line based estimated redshift: {best_redshift:.4f}")
                print(f"Catalog redshift (z_spec): {z_spec:.4f}")

                plt.figure(figsize=(10, 6))
                plt.plot(wavelength_angstrom, flux, lw=1, color='black', label="Observed Flux")
                plt.plot(wavelength_angstrom, flux_baseline_corrected, lw=1, color='grey', alpha=0.5, label="Flux with Baseline (Polynomial Fit)")
                plt.scatter(wavelength_angstrom[peaks], flux[peaks], color='blue', label="Detected Peaks")
                plt.xlabel('Wavelength (Å)', fontsize=14)
                plt.ylabel('Flux (erg/s/cm$^{2}$/Å)', fontsize=14)
                plt.title('1D Spectrum', fontsize=16)

                current_ylim = plt.ylim()
                plt.ylim(current_ylim[0], max(flux) * 1.2)

                for line_name, rest_wavelength in emission_lines.items():
                    obs_wl_est = rest_wavelength * (1 + best_redshift)
                    obs_wl_spec = rest_wavelength * (1 + z_spec)
                    if obs_wl_est <= np.max(wavelength_angstrom):
                        plt.axvline(obs_wl_est, color='red', linestyle='--', alpha=0.8, label="_nolegend_")
                        plt.text(obs_wl_est + 50, max(flux) * 1.05, line_name, rotation=90, verticalalignment='bottom', fontsize=8, color='black')
                        plt.axvline(obs_wl_spec, color='purple', linestyle='--', alpha=0.8, label="_nolegend_")
                        plt.text(obs_wl_spec + 50, max(flux) * 1.1, line_name + " (Spec)", rotation=90, verticalalignment='bottom', fontsize=8, color='blue')
                        plt.axvline(rest_wavelength, color='green', linestyle='--', alpha=0.8, label="_nolegend_")
                        
                current_xlim = plt.xlim()
                plt.xlim(current_xlim[0] - (current_xlim[1] - current_xlim[0]) * 0.05, current_xlim[1])
                custom_legend = [
                    Line2D([0], [0], color='red', linestyle='--', label=f'Estimated (Emission Lines) z={best_redshift:.3f}'),
                    Line2D([0], [0], color='purple', linestyle='--', label=f'Catalog z_spec={z_spec:.3f}'),
                    Line2D([0], [0], color='green', linestyle='--', label='Rest-Frame Lines')
                ]
                plt.legend(handles=custom_legend, title='Redshift Comparison', title_fontproperties={'weight': 'bold'})
                plt.grid(True)
                plt.show()

            else:
                print("Catalog redshift not greater than zero.")
        else:
            print("No matching row in the Excel file for this observation.")
