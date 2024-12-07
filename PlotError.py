from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#fitsFile = "C:\\Development\\Astro_FYP\\HLSP\\hlsp_jades_jwst_nirspec_goods-s-deephst-00002332_clear-prism_v1.0\\hlsp_jades_jwst_nirspec_goods-s-deephst-00002332_clear-prism_v1.0_x1d.fits"
#fitsFile = "C:\\Development\\Astro_FYP\\HLSP\\hlsp_jades_jwst_nirspec_goods-s-deephst-00006438_clear-prism_v1.0\\hlsp_jades_jwst_nirspec_goods-s-deephst-00006438_clear-prism_v1.0_x1d.fits"
#fitsFile = "C:\\Development\\Astro_FYP\\HLSP\\hlsp_jades_jwst_nirspec_goods-s-deephst-00008113_clear-prism_v1.0\\hlsp_jades_jwst_nirspec_goods-s-deephst-00008113_clear-prism_v1.0_x1d.fits"
#fitsFile = "C:\\Development\\Astro_FYP\\HLSP\\hlsp_jades_jwst_nirspec_goods-s-deephst-00017566_clear-prism_v1.0\\hlsp_jades_jwst_nirspec_goods-s-deephst-00017566_clear-prism_v1.0_x1d.fits"

fitsFile = "C:\\Development\\Astro_FYP\\HLSP\\hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0\\hlsp_jades_jwst_nirspec_goods-s-deephst-00018090_clear-prism_v1.0_x1d.fits"

with fits.open(fitsFile) as spec:
    spec.info()
    
    data = spec[1].data
    
    wavelength = data['WAVELENGTH'] 
    flux = data['FLUX']  
    
wavelength = wavelength * 1e4

plt.figure(figsize=(10, 6))
plt.plot(wavelength, flux, label='Flux', lw=1, color='black')
plt.xlabel('Wavelength (Å)', fontsize=14)
plt.ylabel('Flux (erg/s/cm$^{2}$/Å)', fontsize=14)
plt.title('1D Spectrum', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
    