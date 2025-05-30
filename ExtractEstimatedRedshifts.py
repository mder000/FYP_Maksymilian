from astropy.io import fits
import numpy as np
import pandas as pd
import os

#Result of this is provided in Example_Data
fitsFile = "Spectra/jades_dr3_prism_public_gn_v1.1.fits"
#fitsFile = "Spectra/jades_dr3_prism_public_gs_v1.1.fits"
output_excel = "Example_Data/Extracted_observations_FlagA.xlsx"

with fits.open(fitsFile) as spec:
    spec.info()

    # View the columns
    data = spec[1].data
    
    # Print all column names and corresponding values for the chosen observation
    first_observation = data[1]  
    print("First Observation Details:")
    print("=" * 50)
    for col_name in data.columns.names:
        print(f"{col_name}: {first_observation[col_name]}")
    print(len(data.columns))
    
    # Save observations to excel
    nir_id = data['NIRSpec_ID']
    tier = data['TIER']
    z_spec = data['z_Spec']
    z_spec_flag = data['z_Spec_flag']
    
    nir_id_str = [str(id).zfill(8) for id in nir_id]

    # # Create a pandas DataFrame
    df = pd.DataFrame({
        'NIRSpec_ID': nir_id_str,
        'TIER': tier,
        'z_Spec': z_spec,
        'z_Spec_flag': z_spec_flag
    })

    if os.path.exists(output_excel):
        existing_df = pd.read_excel(output_excel, dtype={'NIRSpec_ID': str})  
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(output_excel, index=False)
    
