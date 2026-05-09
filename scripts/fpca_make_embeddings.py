from skfda import FDataGrid

from src.Filter import *
from src.MNEDataPreparation import MNEDataPreparation
from loader.eeg_recording import EEGLoader
from loader.patients import PatientsCSVLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from skfda.representation.basis import FourierBasis, BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA
import pandas as pd
import mne
import umap

from src.PatientsDSsetup import PatientsDSsetup

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

    # Instead of representing my EEG signal with thousands of points, I want to approximate it using only
    # K smooth curves.
    # My signal ≈ a_1·basis_1 + a_2·basis_2 + ... + a_k·basis_k
    K = 4  # output dimensions per channel
    basis = BSplineBasis(n_basis=K)  # = [b_1(t), b_2(t), ..., b_k(t)]
    completed_samples = []  # store the correctly processed samples
    sub_inputs = []
    ep_amt = 0

    for (patient_id, night_id) in samples:

        print(f"ANALYZING {patient_id}, {night_id}")

        try:
            # loading the eeg file from the folder saved locally
            cut = 3000000  # amount of samples to keep
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

        # converting the MNE epochs object into a NumPy array.
        ep_data: np.ndarray = epochs.get_data()
        num_ep = ep_data.shape[0]

        # TODO: Warning, this could be wrong because the documentation at
        #  https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.get_data tells that ep_data has
        #  the shape of [epochs, channels, time] and not [epochs, time, channels]

        for epoch in range(num_ep):
            for channel in range(6):
                sub_inputs.append(ep_data[epoch, channel, :])

    # Converting list of signals into a functional data object.
    # I'm commenting this since I want to be sure to getting the underlying of this:
    # Basically sub_inputs is a list where each element is a np.ndarray containing all the values given a certain
    # epoch and a certain channel.
    # Then FDataGrid change the view by storing that information as a data_matrix =
    # (number of signals = number of epochs * amount of channels,
    # number of observation per signal = length of epoch * frequency,
    # number of values at each point = 1 in our case)
    fd = FDataGrid(sub_inputs)
    basis_fd = fd.to_basis(basis)  # shape of (number_of_signals, K), for each signal it stores [a_1, a_2, ..., a_k]

    # Important note, here FPCA is not performing a dimension reduction, it is just changing the basis to a
    # better representation
    fpca = FPCA(n_components=K,  # components shown at video, no larger than original data
                centering=True,  # Subtract the average curve of the data
                components_basis=basis,  # Spline FPCA
                )
    fpca.fit(basis_fd)

    # After fitting fpca
    # passing from, for each signal,
    # a_1·basis_1 + a_2·basis_2 + ... + a_k·basis_k to
    # mean + c_1·principal_1 + c_2·principal_2 + ... + c_k·principal_k
    # where:
    # - mean function = fpca.mean_
    # - principal components (the new basis) = fpca.components_  [principal_1(t), ..., principal_k(t)]
    # - scores = fpca.transform(basis_fd)  # Shape: (n_signal, K) i.e. for each signal [c_1, c_2, ..., c_k]

    from sklearn.preprocessing import StandardScaler

    scores = fpca.transform(basis_fd)  # (n_signal, K) i.e. for each signal [c_1, c_2, ..., c_k]
    # Standardizing the scores (i.e. new_value = (value - mean) / std)
    # Why do we do that? Because without scaling some features dominate others (some very large values, some very small)
    scores_scaled = StandardScaler().fit_transform(scores)

    # Reconstruct per EEG
    X = []
    idx = 0

    for _ in samples:
        # Number of sub-windows for this EEG
        n_subwindows = ep_amt * 6  # (n_epochs * n_channels)
        # Extract scores for this EEG's sub-windows
        eeg_scores = scores[idx:idx + n_subwindows, :]  # Shape: (n_subwindows, K)
        print(eeg_scores)
        # Concatenate all sub-window scores into a single feature vector
        eeg_feature_vector = eeg_scores.flatten()  # Shape: (n_subwindows * K,)
        X.append(eeg_feature_vector)
        idx += n_subwindows

    # Shape: (n_eeg_recordings, n_subwindows * K) i.e. storing for each recording all the basis for each signal
    # that composes it, all flattened
    X = np.array(X)
    print("FINAL X SHAPE: ", X.shape)
    np.save("C:/Users/tomma/.cache/kagglehub/datasets/yfrite/polysom/versions/3/Embeddings/fpca_embeddings.npy", X)


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
    # ================================================================================================================

    # Loading the CSV about patients, shape: (n_epochs, n_channels * 5)
    df = (PatientsCSVLoader
          .load_dataframe('C:/Users/tomma/.cache/kagglehub/datasets/yfrite/polysom/versions/3/patients.csv'))

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
