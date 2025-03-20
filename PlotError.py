from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

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
fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0_x1d.fits" # z = 4.776
#fitsFile = "hlsp/hlsp_jades_jwst_nirspec_goods-s-deephst-00003892_clear-prism_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst-00003892_clear-prism_v1.0_x1d.fits" # z = 2.227


with fits.open(fitsFile) as spec:
    spec.info()
    
    data = spec[1].data
    print(spec[1].columns)
    
    wavelength = data['WAVELENGTH'] 
    flux = data['FLUX']  
    
wavelength = wavelength * 1e4
print(np.min(wavelength))
print(np.max(wavelength))

plt.figure(figsize=(10, 6))
plt.plot(wavelength, flux, label='Flux', lw=1, color='black')
plt.xlabel('Wavelength (Å)', fontsize=14)
plt.ylabel('Flux (erg/s/cm$^{2}$/Å)', fontsize=14)
plt.title('1D Spectrum', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
    