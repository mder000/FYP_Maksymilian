from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

fitsFile = "Testing_spectra/2D/hlsp_jades_jwst_nirspec_goods-n-mediumhst-00000604_clear-prism_v1.0_s2d.fits" 

with fits.open(fitsFile) as hdul:
    hdul.info()
    
    # Directly access the data from each HDU
    flux2d     = hdul[1].data         # FLUX, shape (674, 27)
    flux_err2d = hdul[2].data         # FLUX_ERR, shape (674, 27)
    wavelength = hdul[3].data         # WAVELENGTH, shape (674,)

# Optionally, compute inverse variance if required:
# (Be cautious of division by zero in real applications)
ivar2d = 1.0 / (flux_err2d**2)

# For example, to display the signal-to-noise (flux * sqrt(ivar)):
signal_to_noise = flux2d * np.sqrt(ivar2d)

fig_2d = plt.figure(figsize=[20, 1.5])
ax3 = fig_2d.add_axes([0.0, 0.0, 0.99, 0.99])
pmap = ax3.imshow(signal_to_noise, origin='lower', interpolation='nearest',
                  vmin=0.01, vmax=4, aspect="auto", cmap="Greys_r")
plt.title("2D Spectrum Signal-to-Noise")
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.colorbar(pmap, ax=ax3, label='S/N')
plt.show()
