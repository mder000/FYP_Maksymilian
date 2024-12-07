from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Replace this with your FITS file path
fitsFile = "C:\\Development\\Astro_FYP\\HLSP\\hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0\\hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0_x1d.fits"

# Open the FITS file and extract wavelength data
with fits.open(fitsFile) as spec:
    data = spec[1].data
    wavelength = data['WAVELENGTH']

# Convert wavelength to Ångstrom
wavelength_angstrom = wavelength * 1e4

# Compute sampling density (differences between consecutive wavelengths)
sampling_density = np.diff(wavelength_angstrom)

# Plot the sampling density
plt.figure(figsize=(10, 6))
plt.plot(wavelength_angstrom[:-1], sampling_density, label='Sampling Density', lw=1, color='blue')
plt.xlabel('Wavelength (Å)', fontsize=14)
plt.ylabel('Sampling Density (Å)', fontsize=14)
plt.title('Sampling Density Across Wavelengths', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()