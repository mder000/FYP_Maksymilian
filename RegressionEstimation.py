import os
import re
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Custom metric for regression accuracy within a tolerance with threshold=2, this metric returns the fraction of predictions that are within 2 redshift units.
def accuracy_within_threshold(threshold=2):
    def metric(y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        correct = tf.cast(diff < threshold, tf.float32)
        return tf.reduce_mean(correct)
    return metric

observationsFolder = "Spectra/1D"
input_excel = "Example_Data/Extracted_observations.xlsx"

# Read the catalog with redshift labels
df_catalog = pd.read_excel(input_excel, dtype={'NIRSpec_ID': str})

# Define a common wavelength grid for all spectra.
common_min = 6000    
common_max = 56000   
n_bins = 2048       
wavelength_grid = np.linspace(common_min, common_max, n_bins)

# Prepare lists to hold input spectra and corresponding redshift labels
X_list = []
y_list = []

# Use only the first 2700 files for training
fits_files = sorted(os.listdir(observationsFolder))
fits_files_train = fits_files[:2700]

for fits_file in fits_files_train:
    # Parse filename to extract tier and observation ID
    fields = re.search(r"hlsp_jades_jwst_nirspec_(.+)-(\d+)_clear-prism", fits_file)
    if fields:
        tier = fields.group(1)
        obs_id = fields.group(2)
        match_row = df_catalog[(df_catalog["NIRSpec_ID"] == obs_id) & (df_catalog["TIER"] == tier)]
        if not match_row.empty:
            # Only use observations with z_Spec_flag == 'A'
            if match_row["z_Spec_flag"].values[0] != 'A':
                continue

            z_spec = match_row["z_Spec"].values[0]
            if z_spec > 0:
                fits_path = os.path.join(observationsFolder, fits_file)
                with fits.open(fits_path) as spec:
                    data = spec[1].data
                    wavelength = data['wavelength'] * 1e4
                    flux = data['flux']

                # Remove NaN values
                valid = ~np.isnan(flux)
                wavelength = wavelength[valid]
                flux = flux[valid]

                # Sort the arrays by wavelength
                sort_idx = np.argsort(wavelength)
                wavelength = wavelength[sort_idx]
                flux = flux[sort_idx]

                # Normalize the flux (min-max normalization)
                flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

                # Interpolate onto the common wavelength grid
                interp_func = interp1d(wavelength, flux_norm, bounds_error=False, fill_value=0)
                flux_resampled = interp_func(wavelength_grid)

                X_list.append(flux_resampled)
                y_list.append(z_spec)

# Convert lists to numpy arrays
X = np.array(X_list)  
y = np.array(y_list)  

# For CNN, add a channel dimension
X = X[..., np.newaxis]

print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

# Split the data into training and testing sets (training on 80% of the 2700 observations)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a moderate 1D CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(n_bins, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    
    tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.GlobalAveragePooling1D(),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)  
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.Huber(),
    metrics=['mae', accuracy_within_threshold(threshold=2)]
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_split=0.1
)

# Evaluate the model on the test set
test_loss, test_mae, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test MAE (Mean Absolute Error): {test_mae:.4f}")
print(f"Test Accuracy (within threshold 2): {test_accuracy:.4f}")

# Save the trained model using the native TF-Keras format
model.save("redshift_estimator_model.keras")
