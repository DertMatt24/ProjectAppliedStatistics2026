import numpy as np

from loader.eeg_recording import EEGLoader
from matplotlib import pyplot as plt
import mne

from pathlib import Path

class MNEDataPreparation:
    """
    This class provides method to prepare the eeg data for a correct and punctual analysis.
    It uses different method, using the library MNE (very useful for analysing eeg data)
    """
    def __init__(self):
        pass

    @staticmethod
    def __read_directory():
        """
        As in the PatientsDSSetup class, this method reads the directory from the NPYDirectory.txt file,
        in order to reach the eeg data.
        :return: File path which contains the folder containing eeg data.
        """
        current = Path(__file__).resolve().parent

        if current.name == "src":
            file_path = current / "noCommitFile" / "NPYDirectory.txt"
        else:
            file_path = current / "src" / "noCommitFile" / "NPYDirectory.txt"

        if not file_path.exists():
            print(f"DEBUG: File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def __correcting_data(data):
        """
        It corrects the data format to make mne and our dataset able to communicate.
        :param data (RAW): Raw data (MNE class).
        :return: polished raw data.
        """
        sfreq = 200  # sampling frequency, given from the dataset 200 Hz
        ch_names = ['Fp1-M2', 'C3-M2', 'O1-M2', 'Fp2-M1', 'C4-M1', 'O2-M1'] # taken from the dataset
        ch_types = ['eeg'] * len(ch_names)

        # Creating info and object Raw
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(data, info)

        # "translating" the channel from our dataset into the ones used by MNE library
        mapping = {
            "Fp1-M2": "Fp1", "C3-M2": "C3", "O1-M2": "O1",
            "Fp2-M1": "Fp2", "C4-M1": "C4", "O2-M1": "O2"
        }
        raw.rename_channels(mapping)
        raw.set_montage('standard_1020')

        return raw



    @staticmethod
    def loading_data(patient_id, night_id):
        """
        This method loads the eeg file from the folder saved locally, and it transforms this file
        into a RAW object (copy), to use it for eeg analysis.

        :param patient_id: id of the patient
        :param night_id: id of the night
        :return: Raw copy of the data, ready to be used by MNE library.
        """
        # finding local file path
        file_path = MNEDataPreparation.__read_directory()
        # loading the npy file into the system
        data = EEGLoader.load_returning_npy(file_path, patient_id, night_id)

        # Using the npy file, we extract the data contained in it and transform them into a RAW object;
        # already able to "communicate" with MNE library.
        raw = MNEDataPreparation.__correcting_data(data)

        r_copy = raw.copy()
        r_copy.apply_function(lambda x: x * 1e-6)

        # Note: it returns a copy of raw data, since MNE modify the data saved in the folder.
        # Since this is a bad behaviour that not only gives error (the data are saved in read mode only),
        # but also if you modify the data you are using for machine learning you are doing something wrong.

        return r_copy

    @staticmethod
    def cleansing_data(raw, l_freq=1.0, h_freq=None, toPlot= False):
        """
        This method removes the noise from the eeg, through the .filter method, and removes noisy channel
        which could compromise our analysis (since the low number of our channels - 6 - typically the method
        finds_bad_eog does not remove any kind of data).
        Note: EOG: Electrooculography (the movement of the eyes, typical of frontal nodes Fp1, Fp2)

        :param raw: Raw data (MNE class)
        :param l_freq:frequency of low cutoff in pass-band filter (default 1.0).
        :param h_freq: frequency of high cutoff in pass-band filter (default None).
        :param toPlot: boolean to represent the data on screen (default False).
        :return: It returns the ICA (Independent Component Analysis) (MNE class)
        """
        raw_notched = raw.copy().notch_filter(freqs=50)
        raw_filtered = raw_notched.copy().filter(l_freq=l_freq, h_freq=h_freq)

        # n_components: between 1 and the number of channels. In our case 6 channels: view documentation
        # random_state: A seed for the NumPy random number generator (RNG).
        # If None (default), the seed will be obtained from the operating system (see RandomState for details),
        # meaning it will most likely produce different output every time this function or method is run.
        # To achieve reproducible results, pass a value here to explicitly initialize the RNG with a
        # defined state.
        # max_iter: Maximum number of iterations during fit.
        # If 'auto', it will set maximum iterations to 1000 for 'fastica' and to 500 for 'infomax' or 'picard'.
        # The actual number of iterations it took ICA.fit() to complete will be stored in the n_iter_
        # attribute.
        # to see other parameters: https://mne.tools/stable/generated/mne.preprocessing.ICA.html
        ica = mne.preprocessing.ICA(n_components=6, random_state=97, max_iter=800)

        # Run the ICA decomposition on raw data.
        ica.fit(raw_filtered)

        # raw_filtered.set_channel_types({'Fp1': 'eog', 'Fp2': 'eog'})
        eog_channels = ['Fp1', 'Fp2']
        eog_idx, scores = ica.find_bads_eog(raw_filtered, ch_name= eog_channels)

        ica.exclude = eog_idx

        if toPlot:
            ica.plot_scores(scores)

            if not ica.exclude:
                ica.plot_components()
                ica.plot_properties(raw_filtered)
            else:
                ica.plot_properties(raw_filtered, picks=ica.exclude)

            plt.show()

        return raw_filtered, ica

    @staticmethod
    def creating_epochs(raw, duration= 30.0, toPlot= False):
        """
        This method partitions the raw data into smaller data - epochs (MNE class)
        :param raw: Raw data (MNE class)
        :param duration: It represents the length of each epoch. (float)
        :param toPlot: boolean to represent the data on screen. (default False)
        :return: It returns the data partitioned into smaller epochs (MNE class)
        """
        # Create an event every 30 seconds
        # id=1 is just a label we give to these markers, it could be changed
        id = 1
        events = mne.make_fixed_length_events(raw, id=id, duration=duration)

        tmin = 0.0
        tmax = duration

        # We subtract 1/sfreq from tmax to ensure the windows don't overlap by exactly one sample
        tmax_adjusted = tmax - 1.0 / raw.info['sfreq']

        epochs = mne.Epochs(
            raw,
            events,
            event_id={'30s_window': id},
            tmin=tmin,
            tmax=tmax_adjusted,
            baseline=None,  # Do not apply baseline correction for continuous partitioning
            preload=True  # Load data into RAM for faster processing
        )

        if toPlot:
            print(epochs)
            # It will say something like: "40 matching events found... all epochs confirmed"

            # To see the segments visually:
            # it shows one epoch, for all the 6 channels
            epochs.plot(n_epochs=1, n_channels=6, scalings='auto')

            epochs.compute_psd(fmin=1,fmax=40)
            epochs.plot_psd()
            plt.show()

        return epochs

def main():
    pass

if __name__ == '__main__':
    main()