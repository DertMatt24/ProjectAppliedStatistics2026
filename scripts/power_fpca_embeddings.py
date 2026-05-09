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

    # Setting the verbosity of the messages to print (DEBUG, INFO, WARNING, ERROR, or CRITICAL)
    mne.set_log_level('ERROR')

    # Filling the samples with the couples of patient and night id
    samples = []
    for patient_id in range(1, 40 + 1):
        for night_id in range(1, 2 + 1):
            samples.append((patient_id, night_id))

    # Removing problematic samples (Totally not understandable)
    samples.remove((8, 1))
    samples.remove((14, 2))

    # patients = np.random.choice(range(1, 40), size=60, replace=False, p=None)
    # samples = [(patient, 1) for patient in patients (1, 1)]

    # Instead of representing my EEG signal with thousands of points, I want to approximate it using only
    # K smooth curves.
    # My signal ≈ a_1·basis_1 + a_2·basis_2 + ... + a_k·basis_k
    K = 12  # output dimensions per channel
    basis = BSplineBasis(n_basis=K)  # = [b_1(t), b_2(t), ..., b_k(t)]
    X = []
    completed_samples = []  # store the correctly processed samples
    sub_inputs = []
    ep_amt = 0

    for (patient_id, night_id) in samples:

        print(f"ANALYZING {patient_id}, {night_id}")

        try:
            # loading the eeg file from the folder saved locally
            cut = 5000000  # amount of samples to keep
            data = MNEDataPreparation.loading_data(patient_id, night_id, cut=cut)
            # applying filters, removing noise, and maybe prepares ICA
            filtered, ica = MNEDataPreparation.cleansing_data(data, do_ica=False)
            # Splits the cleaned EEG into chunks of n seconds.
            n = 60
            epochs = MNEDataPreparation.creating_epochs(filtered, duration=n)

        except Exception:
            continue

        # store the sample correctly processed
        completed_samples.append((patient_id, night_id))
        # amount of epochs for the current sample
        ep_amt = len(epochs)
        print("NUM EPOCHS: ", ep_amt)

        # Compute power spectral density (this tells us how much of each frequency exists)
        spectrum = epochs.compute_psd(fmax=30.0)

        # Extract the data
        psds, frequencies = spectrum.get_data(return_freqs=True)
        print(f"PSD shape: {psds.shape}")  # (n_epochs, n_channels, n_frequencies)
        # This is a temporary comment just to properly visualize and understand what is in psds. Imagine
        # if 2 epochs and 2 channels with 5 frequencies. Each entry represent how strong is that frequency in that
        # signal as combination of epoch and channel
        # psds =
        # [
        #   [   # epoch 0
        #     [0.1, 0.5, 0.2, 0.1, 0.0],   # channel 0
        #     [0.3, 0.2, 0.4, 0.1, 0.0]    # channel 1
        #   ],
        #   [   # epoch 1
        #     [0.2, 0.6, 0.1, 0.0, 0.0],
        #     [0.1, 0.3, 0.5, 0.2, 0.0]
        #   ]
        # ]
        print(f"Freq shape: {frequencies.shape}")  # (n_frequencies,)

        # The bands that we are considering
        # TODO: why are we assuming to discard gamma? Based on data or on scientific texts?
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            # 'gamma': (30, 40)
        }

        # Creating a dictionary with an empty list for each band
        band_powers = {band: [] for band in bands}

        for epoch in range(ep_amt):
            for band_name, (f_low, f_high) in bands.items():
                band_mask = (frequencies >= f_low) & (frequencies <= f_high)
                # Computing the total power of each channel inside a frequency band
                power = trapezoid(psds[epoch, :, band_mask].T, frequencies[band_mask], axis=1)
                band_powers[band_name].append(power)

        # Convert to arrays: (n_epochs, 6 channels)
        for band in band_powers:
            band_powers[band] = np.array(band_powers[band])

        # Now reshape: for each channel, get all 5 bands over time
        # band_powers['alpha'] is (n_epochs, 6)
        # Transpose to (6, n_epochs) per band, then stack bands

        channel_curves = []
        for channel in range(6):
            # Stack all bands for this channel: (5 bands, n_epochs)
            curves = np.array([band_powers[band][epoch, channel]
                               for band in bands
                               for epoch in range(ep_amt)])
            curves = curves.reshape(len(bands), ep_amt)
            # curves[0, :] → delta power over time
            # curves[1, :] → theta power over time
            # curves[2, :] → alpha power over time
            # curves[3, :] → beta power over time
            # For the current channel
            channel_curves.append(curves)

        channel_curves = np.array(channel_curves)  # Collection of the power curves for each channel
        sub_inputs.append(channel_curves)
        # channel_curves[0] is (5 bands, n_epochs) for channel 0
        # channel_curves[1] is (5 bands, n_epochs) for channel 1, etc.

        print("Channel curves: ", channel_curves[0].shape)

    sub_inputs = np.array(sub_inputs)
    all_curves = []
    indices = []  # Track which recording/channel/band each curve belongs to

    for recording in range(len(completed_samples)):
        for channel in range(6):
            for band in range(4):
                curve = sub_inputs[recording, channel, band, :]
                all_curves.append(curve)
                indices.append((recording, channel, band))

    fd = FDataGrid(all_curves)

    basis_fd = fd.to_basis(basis)
    fpca = FPCA(n_components=K,  # components shown at video, no larger than original data
                centering=True,  # Subtract the average curve of the data
                components_basis=basis,  # Spline FPCA
                )

    fpca.fit(basis_fd)

    # After fitting fpca each signal:
    # mean + c_1·principal_1 + c_2·principal_2 + ... + c_k·principal_k
    # where:
    # - mean function = fpca.mean_
    # - principal components (the new basis) = fpca.components_  [principal_1(t), ..., principal_k(t)]
    # - scores = fpca.transform(basis_fd)  # Shape: (n_signal, K) i.e. for each signal [c_1, c_2, ..., c_k]
    # scores = fpca.transform(basis_fd)  # Shape: (n_subwindows, K)
    from sklearn.preprocessing import StandardScaler

    scores = fpca.transform(basis_fd)
    # Standardizing the scores (i.e. new_value = (value - mean) / std)
    # Why do we do that? Because without scaling some features dominate others (some very large values, some very small)
    scores_scaled = StandardScaler().fit_transform(scores)
    # scores_scaled.shape = (n_curves, K) where n_curves = n_recordings × 6 channels × 4 bands

    # Reconstruct per EEG
    X = []
    # Recompose per recording
    embeddings = []
    # For each recording:
    for recording in range(len(completed_samples)):
        # Indices stores [(recording, channel, band)]
        rec_embedding = scores_scaled[
            [i for i, (r, c, b) in enumerate(indices) if r == recording]
        ].flatten()
        X.append(rec_embedding)

    X = np.array(X)
    print("SHAPE X", X.shape)
    np.save(f"C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/fpca_of_power_embeddings.npy", X)
    import umap
    # non-linear dimensionality reduction method, passes from X.shape = (n_recordings, huge_dimension) to
    # X_2d = (n_recordings, n_components)
    n_components = 2
    reducer = umap.UMAP(n_components=n_components)
    X_2d = reducer.fit_transform(np.array(X))

    # PCA instead of UMAP
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X)

    #  plt.scatter(X_2d[:, 0], X_2d[:, 1])
    # plt.show()

    # ==================================== Plotting Section ==========================================================
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
    for (patient_id, night_id) in completed_samples:
        # Query the dataframe for this patient/night pair
        row = df[(df['user_id'] == patient_id) & (df['night_id'] == night_id)]
        # Store the number of attacks
        attacks = row.iloc[0]['NAp']
        # Handle empty/NaN values
        if pd.isna(attacks) or attacks == '' or attacks == 'NaN':
            attacks = 0
        else:
            attacks = int(attacks)  # Convert to int if it's a string
        attack_counts.append(attacks)
    attack_counts = np.array(attack_counts)
    print("Attacks counts: ", attack_counts)

    # ==================================== Plotting Section ==========================================================
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
    # ================================================================================================================
