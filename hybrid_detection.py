import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------
# Generate synthetic data (normal + anomalies)
# ---------------------------------------------------------
np.random.seed(42)

normal_data = np.random.normal(0, 1, (1000, 10))
anomaly_data = np.random.normal(5, 1, (50, 10))
X_total = np.vstack([normal_data, anomaly_data])

# ---------------------------------------------------------
# Build Autoencoder
# ---------------------------------------------------------
input_dim = X_total.shape[1]
encoding_dim = 5

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='linear')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

# Train autoencoder
autoencoder.fit(X_total, X_total, epochs=20, batch_size=32, verbose=0)

# ---------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------
iso = IsolationForest(contamination=0.05)
iso_labels = iso.fit_predict(X_total)

# ---------------------------------------------------------
# Reconstruction Error
# ---------------------------------------------------------
recon = autoencoder.predict(X_total)
recon_error = np.mean((X_total - recon)**2, axis=1)

threshold = np.percentile(recon_error, 95)
ae_labels = (recon_error > threshold).astype(int)

# ---------------------------------------------------------
# Hybrid Score
# ---------------------------------------------------------
hybrid_labels = []
for iso_label, ae_label in zip(iso_labels, ae_labels):
    if iso_label == -1 or ae_label == 1:
        hybrid_labels.append(1)  # anomaly
    else:
        hybrid_labels.append(0)  # normal

print("Hybrid anomaly detection complete.")
print(f"Total anomalies detected: {sum(hybrid_labels)}")
