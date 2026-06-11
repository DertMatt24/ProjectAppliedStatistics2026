from skfda import FDataGrid

from src.Filter import *
from src.MNEDataPreparation import MNEDataPreparation
from loader.eeg_recording import EEGLoader
from loader.patients import PatientsCSVLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.animation as animation

from skfda.representation.basis import FourierBasis, BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA
import pandas as pd
import mne

from src.PatientsDSsetup import PatientsDSsetup

if __name__ == '__main__':

    mne.set_log_level('ERROR')
    completed_samples = []
    for i in range(1, 40 + 1):
        for j in range(1, 2 + 1):
            completed_samples.append((i, j))

    completed_samples.remove((8, 1))
    completed_samples.remove((14, 2))

    # patients = np.random.choice(range(1, 40), size=60, replace=False, p=None)

   # samples =
    [
        #    (patient, 1) for patient in patients
        (1, 1), (2, 2)
    ]

    X = []
    K = 4  # output dimensions per channel
    basis = BSplineBasis(n_basis=K)

    sub_inputs = []
    ep_amt = 0
    samples = completed_samples

    # completed_samples = []
    for (p_id, n_id) in samples:

        break
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
        spectrum = epochs.compute_psd(fmax=40.0)

        # Extract the data
        psds, freqs = spectrum.get_data(return_freqs=True)

        print(f"PSD shape: {psds.shape}")  # (n_epochs, n_channels, n_freqs)
        print(f"Freq shape: {freqs.shape}")  # (n_freqs,)

        # eeg = EEGLoader.load('C:/Users/picul/Videos/Applied/Dataset/', 1, 1)
        # print(eeg.data)

        last_shape = psds.shape[2]

        # psds shape: (n_epochs, n_channels, n_freqs)
        # freqs shape: (n_freqs,)

        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 40)
        }

        band_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            # Get frequency indices in band
            band_mask = (freqs >= f_low) & (freqs <= f_high)

            # Integrate (trapz for proper integration, or just mean)
            from scipy.integrate import trapezoid

            band_power = trapezoid(psds[:, :, band_mask], freqs[band_mask], axis=2)
            # shape: (n_epochs, n_channels)

            band_powers[band_name] = band_power

        for element in band_powers.values():
            print("Shape of element: ", element.shape)

        result = np.concatenate([band_powers[k] for k in sorted(band_powers.keys())], axis=1)

        # Combine into feature matrix
        print("FEATURES SHAPE:", result.shape)
        sub_inputs.append(result)

        # ep_data: np.ndarray = epochs.get_data()
        # num_ep = ep_data.shape[0]

        # for ep in range(num_ep):
        #     for ch in range(6):
        #         sub_inputs.append(ep_data[ep, :, ch])

        """
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
        """
    # X = np.array(sub_inputs)

    # np.save("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy", X)
    X = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/standard_embeddings.npy")
    # X = X.reshape(X.shape[0], -1)

    spectral_centroids = []

    for example in []:
        # example shape: (num_windows, 30)
        num_windows = example.shape[0]
        n_bands = 5
        n_channels = 6
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        freqs = np.array([2, 6, 10, 20, 40])

        channel_centroids = np.zeros((n_channels, num_windows))

        for ch in range(n_channels):
            # Extract power for this channel across all bands and windows
            power_all_bands = example[:, ch::n_channels]  # (num_windows, n_bands)

            # Apply spectral_centroid
            centroid = np.sum(power_all_bands * freqs, axis=1) / np.sum(power_all_bands, axis=1)

            window_size = 3
            filtered = np.convolve(centroid, np.ones(window_size) / window_size, mode='same')
            channel_centroids[ch, :] = filtered

        spectral_centroids.append(channel_centroids)

    spectral_centroids = np.array(spectral_centroids)
    np.save("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/spectral_centroids.npy", spectral_centroids)

    # Shape: (num_examples, num_channels, num_windows)

    print("Shape: ", spectral_centroids.shape)
    # plt.plot(spectral_centroids[0].T)

    # plt.show()


    print(X.shape)

    X = X[60]
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
    # for i, (p_id, n_id) in enumerate(completed_samples):
        #plt.annotate(f"{p_id}_{n_id}", (X_2d[i, 0], X_2d[i, 1]), fontsize=8)


    for i in range(X.shape[0]):
        plt.annotate(f"{i}", (X_2d[i, 0], X_2d[i, 1]), fontsize=8)


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

    print(attack_counts)
    attack_counts = np.array(attack_counts)

    attack_counts = np.arange(X.shape[0])
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

    windows_per_90min = 90  # 90 minutes * 60 seconds / 60 seconds per window

    cycle_indices = np.arange(X.shape[0]) // windows_per_90min

    num_cycles = int(np.ceil(X.shape[0] / windows_per_90min))

    colors = plt.cm.tab20(np.linspace(0, 1, num_cycles))

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=cycle_indices,
                         cmap='tab20',
                         s=100,
                         alpha=0.7,
                         edgecolors='k',
                         linewidth=0.5)

    cbar = plt.colorbar(scatter, label='Sleep Cycle (90 min intervals)', ax=ax)
    ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
    ax.set_title('UMAP of EEG Features (colored by 90-min sleep cycles)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    for i in range(X.shape[0]):
        cycle = cycle_indices[i]
        ax.annotate(f"{i}", (X_2d[i, 0], X_2d[i, 1]), fontsize=7, alpha=0.6)

    plt.tight_layout()
    plt.show()

    print(f"Total windows: {X.shape[0]}")
    print(f"Windows per 90-min cycle: {windows_per_90min}")
    print(f"Number of complete cycles: {num_cycles}")
    print(f"Cycle distribution: {np.bincount(cycle_indices)}")

    import numpy as np
    import matplotlib.pyplot as plt
    import skfuzzy as fuzz

    # Fuzzy C-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_2d.T,  # shape must be (features, samples)
        c=4,
        m=2.0,
        error=1e-5,
        maxiter=1000,
        seed=42
    )

    # Membership to cluster 1
    cluster_indices = np.arange(u.shape[0])  # [0, 1, 2, 3]

    # Weighted average across cluster indices
    membership_continuous = (u * cluster_indices[:, np.newaxis]).sum(axis=0)
    membership = membership_continuous
    # Time axis
    t = np.arange(len(membership))

    fig, ax = plt.subplots(figsize=(14, 4))

    from scipy.signal import medfilt

    # binary = (membership > 0.5).astype(int)

    membership = medfilt(membership, kernel_size=5)
    ax.plot(t, membership, linewidth=1.5, color='tab:blue')
    ax.scatter(
        t,
        membership,
        c=membership,
        cmap='coolwarm',
        s=15
    )

    # 90-window boundaries
    for x in range(0, len(t), 90):
        ax.axvline(x, color='k', linestyle='--', alpha=0.2)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Window')
    ax.set_ylabel('Membership')
    ax.set_title('Fuzzy C-Means Membership Over Time')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Fuzzy Partition Coefficient (FPC): {fpc:.3f}")


    windows_per_90min = 90
    cycle_indices = np.arange(X.shape[0]) // windows_per_90min

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter([], [], c=[], cmap='tab20', s=100, alpha=0.7, edgecolors='k', linewidth=0.5, vmin=0,
                         vmax=cycle_indices.max())
    line, = ax.plot([], [], 'k-', alpha=0.3, linewidth=1.5)

    ax.set_xlim(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1)
    ax.set_ylim(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1)
    ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
    ax.set_title('UMAP Trajectory: EEG Features Over Time', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, label='Sleep Cycle (90 min)', ax=ax)

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


    def animate(frame):
        n_points = frame + 1

        scatter.set_offsets(X_2d[:n_points])
        scatter.set_array(cycle_indices[:n_points])

        line.set_data(X_2d[:n_points, 0], X_2d[:n_points, 1])

        window_num = frame
        minutes_elapsed = (window_num * 60) / 60
        cycle_num = cycle_indices[frame]

        time_text.set_text(f'Window: {window_num}\nTime: {minutes_elapsed:.1f} min\nCycle: {cycle_num}')

        return scatter, line, time_text


    anim = animation.FuncAnimation(fig, animate, frames=X.shape[0],
                                   interval=50, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()

    # anim.save('umap_animation.mp4', writer='ffmpeg', fps=20, dpi=100)


    windows_per_cycle = 70

    total_windows = X.shape[0]
    num_cycles = int(np.ceil(total_windows / windows_per_cycle))

    cycle_indices = np.arange(total_windows) // windows_per_cycle

    padded_size = num_cycles * windows_per_cycle
    X_2d_padded = np.vstack([X_2d, np.zeros((padded_size - total_windows, 2))])
    cycle_indices_padded = np.concatenate([cycle_indices, np.full(padded_size - total_windows, -1)])

    ncols = 3
    nrows = int(np.ceil(num_cycles / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    global_min_x, global_max_x = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    global_min_y, global_max_y = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    for cycle_num in range(num_cycles):
        ax = axes[cycle_num]

        start_idx = cycle_num * windows_per_cycle
        end_idx = start_idx + windows_per_cycle

        cycle_data = X_2d_padded[start_idx:end_idx]
        cycle_labels = cycle_indices_padded[start_idx:end_idx]

        valid_mask = cycle_labels >= 0
        valid_data = cycle_data[valid_mask]
        valid_labels = cycle_labels[valid_mask]

        scatter = ax.scatter(cycle_data[:, 0], cycle_data[:, 1],
                             c=np.arange(windows_per_cycle),
                             cmap='twilight',
                             s=60,
                             alpha=0.6,
                             edgecolors='k',
                             linewidth=0.5)

        if len(valid_data) > 1:
            ax.plot(valid_data[:, 0], valid_data[:, 1], 'k-', alpha=0.2, linewidth=1.5)

        ax.set_xlim(global_min_x, global_max_x)
        ax.set_ylim(global_min_y, global_max_y)
        ax.set_title(f'Cycle {cycle_num} ({start_idx}-{end_idx - 1})', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)

        if valid_mask.sum() < windows_per_cycle:
            ax.text(0.5, 0.05, f'({int(valid_mask.sum())}/{windows_per_cycle} windows)',
                    transform=ax.transAxes, ha='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    for idx in range(num_cycles, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'UMAP Progression by Sleep Cycles (windows_per_cycle={windows_per_cycle})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

    print(f"Total windows: {total_windows}")
    print(f"Windows per cycle: {windows_per_cycle}")
    print(f"Number of cycles: {num_cycles}")

    windows_per_cycle = 110

    total_windows = X.shape[0]
    num_cycles = int(np.ceil(total_windows / windows_per_cycle))

    cycle_indices = np.arange(total_windows) // windows_per_cycle

    padded_size = num_cycles * windows_per_cycle
    X_2d_padded = np.vstack([X_2d, np.zeros((padded_size - total_windows, 2))])
    cycle_indices_padded = np.concatenate([cycle_indices, np.full(padded_size - total_windows, -1)])

    ncols = 3
    nrows = int(np.ceil(num_cycles / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    global_min_x, global_max_x = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    global_min_y, global_max_y = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    scatters = []
    lines = []
    time_texts = []

    for cycle_num in range(num_cycles):
        ax = axes[cycle_num]

        scatter = ax.scatter([], [], cmap='twilight', s=60, alpha=0.6, edgecolors='k', linewidth=0.5)
        line, = ax.plot([], [], 'k-', alpha=0.2, linewidth=1.5)
        time_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlim(global_min_x, global_max_x)
        ax.set_ylim(global_min_y, global_max_y)
        ax.set_title(f'Cycle {cycle_num}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)

        scatters.append(scatter)
        lines.append(line)
        time_texts.append(time_text)

    for idx in range(num_cycles, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'UMAP Progression by Sleep Cycles (windows_per_cycle={windows_per_cycle})',
                 fontsize=14, fontweight='bold', y=0.995)


    def animate(frame):
        points_to_show = frame + 1

        for cycle_num in range(num_cycles):

            start_idx = cycle_num * windows_per_cycle
            end_idx = min(
                start_idx + points_to_show,
                total_windows,
                (cycle_num + 1) * windows_per_cycle
            )

            if end_idx > start_idx:

                cycle_data = X_2d[start_idx:end_idx]

                colors = np.arange(len(cycle_data))

                scatters[cycle_num].set_offsets(cycle_data)
                scatters[cycle_num].set_array(colors)
                scatters[cycle_num].set_clim(0, windows_per_cycle)

                if len(cycle_data) > 1:
                    lines[cycle_num].set_data(
                        cycle_data[:, 0],
                        cycle_data[:, 1]
                    )

                time_texts[cycle_num].set_text(
                    f'{len(cycle_data)}/{windows_per_cycle}'
                )

            else:
                scatters[cycle_num].set_offsets(np.empty((0, 2)))
                lines[cycle_num].set_data([], [])
                time_texts[cycle_num].set_text('0/0')

        return scatters + lines + time_texts

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=windows_per_cycle,
        interval=50,
        blit=True,
        repeat=True
    )
    plt.tight_layout()
    plt.show()

    # anim.save('umap_multiplot_animation.mp4', writer='ffmpeg', fps=20, dpi=100)