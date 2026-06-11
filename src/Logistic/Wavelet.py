import numpy as np
import pandas as pd
import pywt
import mne
from pathlib import Path

from src.MNEDataPreparation import MNEDataPreparation


def wavelet_features_from_signal(signal, wavelet="db4", level=5):
    """
    signal: 1D EEG epoch for one channel
    returns: features from A5, D5, D4, D3
    """
    # Extract A5, D5, D4, D3, D2, D1 for one epoch for one channel
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    A5, D5, D4, D3, D2, D1 = coeffs

    selected = {
        "delta_A5": A5,
        "theta_D5": D5,
        "alpha_D4": D4,
        "beta_D3": D3
    }

    features = []

    # for each coefficient of the selected (epoch, channel), compute some metrics to capture information
    for name, c in selected.items():
        # Now I compute energy, standard deviation and mean to capture information
        energy = np.sum(c ** 2)
        mean_abs = np.mean(np.abs(c))
        std = np.std(c)

        features.extend([energy, mean_abs, std])

    return np.array(features)


def wavelet_features_from_epochs(epochs, wavelet="db4", level=5):

    # data [epoch, channel, samples]
    data = epochs.get_data()

    epoch_features = []

    # for each epoch:
    for ep in range(data.shape[0]):
        channel_features = []
        # for each channel:
        for ch in range(data.shape[1]):
            # All the sampling in every t for a specific epoch (time window) and channel
            signal = data[ep, ch, :]
            # Returns the selected metrics for the selected epoch
            feats = wavelet_features_from_signal(
                signal,
                wavelet=wavelet,
                level=level
            )
            # Add the metrics in the channel features
            channel_features.extend(feats)

        # Append all the channels features of the selected epoch
        epoch_features.append(channel_features)

    epoch_features = np.array(epoch_features)

    # For each night we have:
    #               feature_1        feature_2        feature_3      ...
    # epoch 1       10               3                20
    # epoch 2       12               4                18
    # epoch 3       8                2                25
    # epoch 4       11               3                19
    #
    #
    # Where
    # feature_1  = energy(A5) channel 1
    # feature_2  = std(A5)    channel 1
    #
    # feature_3  = energy(D5) channel 1
    # feature_4  = std(D5)    channel 1
    #
    # ...
    #
    # feature_8  = std(D3)    channel 1
    # feature_9  = energy(A5) channel 2
    # feature_10 = std(A5)    channel 2
    # We compute the mean of each feature along all the epochs of the same night
    night_mean = epoch_features.mean(axis=0)
    night_std = epoch_features.std(axis=0)
    night_embedding = np.concatenate([night_mean, night_std])

    return night_embedding, epoch_features


def main():

    mne.set_log_level("ERROR")

    # Build a list of tuple (patient_id, night_id)
    samples = []
    for patient_id in range(1, 41):
        for night_id in range(1, 3):
            samples.append((patient_id, night_id))
    # Remove the problematic samples
    samples.remove((8, 1))
    samples.remove((14, 2))

    X = [] # data structure that will contain all the embeddings
    Y = []
    completed_samples = []
    # For each (patient_id, night_id)
    for patient_id, night_id in samples:
        print(f"Processing patient {patient_id}, night {night_id}")

        try:
            #  Loads the eeg file from the folder saved locally
            raw = MNEDataPreparation.loading_data(
                patient_id,
                night_id,
                cut=5_000_000
            )

            # Removing noise
            filtered, _ = MNEDataPreparation.cleansing_data(
                raw,
                do_ica=False
            )

            # Creating epochs of 60 seconds
            epochs = MNEDataPreparation.creating_epochs(
                filtered,
                duration=60
            )

            # Creating an embedding based on wavelet.
            # To extract information we store mean and std
            embedding, inter_embedding = wavelet_features_from_epochs(
                epochs,
                wavelet="db4",
                level=5
            )

            X.append(embedding)
            Y.append(inter_embedding)
            # Samples embedded
            completed_samples.append((patient_id, night_id))

        except Exception as e:
            print(f"Skipped ({patient_id}, {night_id}) because: {e}")

    # Transform it into a numpy array
    X = np.array(X)
    Y = np.array(Y)
    print("Final embedding shape:", X.shape)
    print("Completed samples:", len(completed_samples))

    # Save data into files
    np.save("wavelet_embeddings_avg", X)
    np.save("wavelet_completed_samples", np.array(completed_samples))
    np.save("standard_embeddings", Y)



if __name__ == "__main__":
    main()