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

from src.PatientsDSsetup import PatientsDSsetup

if __name__ == '__main__':

    mne.set_log_level('ERROR')
    samples = []
    for i in range(1, 40):
        for j in range(1, 2 + 1):
            samples.append((i, j))

    samples.remove((8, 1))
    samples.remove((14, 2))

    # patients = np.random.choice(range(1, 40), size=60, replace=False, p=None)
    X = []
    K = 4  # output dimensions per channel
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


        ep_data: np.ndarray = epochs.get_data()
        num_ep = ep_data.shape[0]

        for ep in range(num_ep):
            for ch in range(6):
                sub_inputs.append(ep_data[ep, :, ch])


    fd = FDataGrid(sub_inputs)

    basis_fd = fd.to_basis(basis)
    fpca = FPCA(n_components=K,  # components shown at video, no larger than original data
                centering=True,  # Subtract the average curve of the data
                components_basis=basis,  # Fourier FPCA
    )

    fpca.fit(basis_fd)

    # After fitting fpca
    # scores = fpca.transform(basis_fd)  # Shape: (n_subwindows, K)
    from sklearn.preprocessing import StandardScaler

    scores = fpca.transform(basis_fd)
    scores_scaled = StandardScaler().fit_transform(scores)

    # Reconstruct per EEG
    X = []

    idx = 0
    for (p_id, n_id) in samples:
        # Number of sub-windows for this EEG
        n_subwindows = ep_amt * 6  # (n_epochs * n_channels)

        # Extract scores for this EEG's sub-windows
        eeg_scores = scores[idx:idx + n_subwindows, :]  # Shape: (n_subwindows, K)

        print(eeg_scores)
        # Concatenate all sub-window scores into a single feature vector
        eeg_feature_vector = eeg_scores.flatten()  # Shape: (n_subwindows * K,)

        X.append(eeg_feature_vector)

        idx += n_subwindows

    X = np.array(X)  # Shape: (n_eeg_recordings, n_subwindows * K)
    print("FINAL X SHAPE: ", X.shape)
    np.save("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/fpca_embeddings.npy", X)
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
