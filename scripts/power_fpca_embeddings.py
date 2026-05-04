from skfda import FDataGrid
from scipy.integrate import trapezoid

# from src.Filter import *
from src.MNEDataPreparation import MNEDataPreparation
# from loader.eeg_recording import EEGLoader
from loader.patients import PatientsCSVLoader
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
import numpy as np
from skfda.representation.basis import FourierBasis, BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA
import pandas as pd
import mne

# from src.PatientsDSsetup import PatientsDSsetup

if __name__ == '__main__':

    mne.set_log_level('ERROR')
    samples = []
    for i in range(1, 40):
        for j in range(1, 2 + 1):
            samples.append((i, j))

    samples.remove((8, 1))
    samples.remove((14, 2))

    # patients = np.random.choice(range(1, 40), size=60, replace=False, p=None)

    #samples = [
    #    (patient, 1) for patient in patients
    (1, 1)
    #]

    X = []
    K = 12 # output dimensions per channel
    basis = BSplineBasis(n_basis=K)

    sub_inputs = []
    ep_amt = 0
    completed_samples = []
    for (p_id, n_id) in samples:

        print(f"ANALYZING {p_id}, {n_id}")

        try:
            data = MNEDataPreparation.loading_data(p_id, n_id, cut=5000000)
            filtered, ica = MNEDataPreparation.cleansing_data(data, do_ica=False)
            epochs = MNEDataPreparation.creating_epochs(filtered, 60)
        except Exception:
            continue
        completed_samples.append((p_id, n_id))
        ep_amt = len(epochs)
        # Compute power spectral density
        print("NUM EPOCHS: ", ep_amt)
        spectrum = epochs.compute_psd(fmax=30.0)

        # Extract the data
        psds, freqs = spectrum.get_data(return_freqs=True)

        print(f"PSD shape: {psds.shape}")  # (n_epochs, n_channels, n_freqs)
        print(f"Freq shape: {freqs.shape}")  # (n_freqs,)
        # psds shape: (n_epochs, n_channels, n_freqs)
        # freqs shape: (n_freqs,)

        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            # 'gamma': (30, 40)
        }

        band_powers = {band: [] for band in bands}

        for epoch in range(ep_amt):
            for band_name, (f_low, f_high) in bands.items():
                band_mask = (freqs >= f_low) & (freqs <= f_high)
                power = trapezoid(psds[epoch, :, band_mask].T, freqs[band_mask], axis=1)
                band_powers[band_name].append(power)

        # Convert to arrays: (n_epochs, 6 channels)
        for band in band_powers:
            band_powers[band] = np.array(band_powers[band])

        # Now reshape: for each channel, get all 5 bands over time
        # band_powers['alpha'] is (n_epochs, 6)
        # Transpose to (6, n_epochs) per band, then stack bands

        channel_curves = []
        for ch in range(6):
            # Stack all bands for this channel: (5 bands, n_epochs)
            curves = np.array([band_powers[band][epoch, ch] for band in bands for epoch in range(ep_amt)])
            curves = curves.reshape(len(bands), ep_amt)
            channel_curves.append(curves)

        channel_curves = np.array(channel_curves)
        sub_inputs.append(channel_curves)
        # channel_curves[0] is (5 bands, n_epochs) for channel 0
        # channel_curves[1] is (5 bands, n_epochs) for channel 1, etc.

        print("Channel curves: ", channel_curves[0].shape)

    sub_inputs = np.array(sub_inputs)
    all_curves = []
    indices = []  # Track which recording/channel/band each curve belongs to

    for rec in range(len(completed_samples)):
        for ch in range(6):
            for band in range(4):
                curve = sub_inputs[rec, ch, band, :]
                all_curves.append(curve)
                indices.append((rec, ch, band))

    fd = FDataGrid(all_curves)

    basis_fd = fd.to_basis(basis)
    fpca = FPCA(n_components=K,  # components shown at video, no larger than original data
                centering=True,  # Subtract the average curve of the data
        components_basis=basis,  # Spline FPCA
    )

    fpca.fit(basis_fd)

    # After fitting fpca
    # scores = fpca.transform(basis_fd)  # Shape: (n_subwindows, K)
    from sklearn.preprocessing import StandardScaler

    scores = fpca.transform(basis_fd)
    scores_scaled = StandardScaler().fit_transform(scores)

    # Reconstruct per EEG
    X = []
    # Recompose per recording
    embeddings = []
    for rec in range(len(completed_samples)):
        rec_embedding = scores_scaled[
            [i for i, (r, c, b) in enumerate(indices) if r == rec]
        ].flatten()
        X.append(rec_embedding)

    X = np.array(X)
    print("SHAPE X", X.shape)
    np.save(f"C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/fpca_of_power_embeddings.npy", X)
    import umap

    reducer = umap.UMAP(n_components=2)
    X_2d = reducer.fit_transform(np.array(X))

    # PCA instead of UMAP
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X)

    #  plt.scatter(X_2d[:, 0], X_2d[:, 1])
    # plt.show()

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1])

    print("Shape: ", X_2d.shape)
    for i, (p_id, n_id) in enumerate(completed_samples):
        plt.annotate(f"{p_id}_{n_id}", (X_2d[i, 0], X_2d[i, 1]), fontsize=8)

    plt.show()
    df = PatientsCSVLoader.load_dataframe('C:/Users/picul/Videos/Applied/Dataset_Full/patients.csv')
    # shape: (n_epochs, n_channels * 5)

    # Extract attack counts for each sample
    attack_counts = []
    for (p_id, n_id) in completed_samples:
        # Query the dataframe for this patient/night pair
        row = df[(df['user_id'] == p_id) & (df['night_id'] == n_id)]

        attacks = row.iloc[0]['NAp']

        # Handle empty/NaN values
        if pd.isna(attacks) or attacks == '' or attacks == 'NaN':
            attacks = 0
        else:
            attacks = int(attacks)  # Convert to int if it's a string

        attack_counts.append(attacks)

    attack_counts = np.array(attack_counts)

    print("Attacks counts: ", attack_counts)

    # Plot with green-to-red colormap
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=attack_counts,
                          cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red=high)
                          s=100,
                          vmin=0,
                          vmax=attack_counts.max())

    cbar = plt.colorbar(scatter, label='Number of Attacks')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP of EEG FPCA Features (colored by attacks)')
    plt.show()
    # Reshape and average
