import os
from astroquery.mast import Observations
import pandas as pd
from astropy.table import Table

save_directory = "jades_dr1_spectra"
os.makedirs(save_directory, exist_ok=True)

obs = Observations.query_criteria(
    provenance_name="jades",
    instrument_name="NIRSpec",
    filters="PRISM"
)

data_products = Observations.get_product_list(obs)

data_products_df = data_products.to_pandas()

filtered_products = data_products_df[
    data_products_df['productFilename'].str.contains(r'_x1d\.fits')
]

if not filtered_products.empty:
    filtered_products_table = Table.from_pandas(filtered_products)
    
    downloaded_files = Observations.download_products(
        filtered_products_table,
        download_dir=save_directory
    )
    print(f"Downloaded files are saved in {save_directory}")
else:
    print("No matching files found for the specified filters.")