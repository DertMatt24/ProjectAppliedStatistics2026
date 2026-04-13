from os.path import join, isdir, isfile
from pathlib import Path
from typing import Union, Tuple
from warnings import warn

import numpy as np
from scipy.signal import resample

class EEGRecording:
    """ The main class to store and handle raw EEGRecordings for visualization before
    converting it into some format suitable for machine learning models. This class allows
    storage of the eeg, its raw data, its frequency, its time interval. It makes operations
    easier by providing a light wrapper on the underlying numpy array which keeps track of the
    patient id, the night id, frequency and the time the array represents.

    This also allows to extract cheap intervals from the EEGRecording, as numpy avoids creating
    copies instead making views of the original data when indexing contiguous memory.
    """

    def __init__(self, frequency: int, data: np.ndarray, t0: float = 0.0):
        """ Generate a new EEGRecording object with given frequency.

        :param frequency: the frequency of the EEGRecording
        :param data: the raw numpy array from the recording
        :param t0: the initial time of the data (as an offset to the beginning of the recording,
            NOT the actual clock time)
        """
        self.frequency = frequency
        self.data = data

        self.patient_id = None
        self.night_id = None

        self.begin_time = t0
        self.end_time = self.begin_time + self.time_duration()

        # The user may want to access the parent EEG from a view... better to have it
        # that not honestly
        self.parent = None

    def give_ids(self, patient_id: int, night_id: int):
        """ Set the patient and night id for a recording. """
        self.patient_id = patient_id
        self.night_id = night_id

    def downsample(self, new_freq: int) -> Union['EEGRecording', None]:
        """ Generate a new recording with a lower frequency.
        NOTE: This may affect the quality of the data! (Nyquist sampling frequency). In general:
        - Delta band c.a. 8hz required
        - Theta band c.a. 16hz required
        - Alpha band c.a. 24hz required
        - Beta band c.a. 60hz required
        - Gamma band c.a. 200hz required

        :return: a downsampled EEG.
        """
        if new_freq > self.frequency:
            warn("Cannot resample the EEGRecording to a higher frequency")
            return None
        curr_num = np.size(self.data) / self.channels()
        # A number betweeon 0 and 1
        frequency_ratio = new_freq / self.frequency
        num = int(frequency_ratio * curr_num)
        downsampled_data = resample(self.data, num, t=None, axis=0, window=None, domain='time')

        eeg_obj = EEGRecording(new_freq, downsampled_data, t0=self.begin_time)
        eeg_obj.give_ids(self.patient_id, self.night_id)
        eeg_obj.parent = self
        return eeg_obj

    def trim_from_timestamps(self, time_0: float, time_1: float) -> Union['EEGRecording', None]:
        """ Cut the EEG recording and generate a new EEGRecording object with the data selected from time_0
        to time_1 (inclusive, if the frequency allows the precision).
        The new object has the current object as a parent, its data is a view of the current data starting from the
        given time and ending at the final time

        :param time_0: The beginning of the time interval
        :param time_1: The end of the time interval
        :return: a new EEGRecording object.
        """
        if time_0 < self.begin_time or time_1 > self.end_time:
            warn("Attempting to create a subview of an EEG recording that exceeds the end time or beginning time")
            return None
        if time_0 > time_1:
            warn("Attempting to create a subview of an EEG with initial time greater than final time.")
            return None
        low_index = max(0, int((time_0 - self.begin_time)*self.frequency))
        upper_index = min(int((time_1 - self.begin_time)*self.frequency) - self.end_time, self.num_samples()-1)
        # This creates a numpy view, no new memory as long as the array is not modified
        data_view = self.data[low_index:upper_index]

        eeg_obj = EEGRecording(self.frequency, data_view, t0=time_0)
        eeg_obj.give_ids(self.patient_id, self.night_id)
        eeg_obj.parent = self
        return eeg_obj

    def raw_data(self) -> np.ndarray:
        """ Get the underlying raw data"""
        return self.data

    def timestamps_array(self) -> np.ndarray:
        """ Return a numpy array where each i-th entry is the time stamp of the i-th sample """
        return np.arange(self.num_samples()) / self.frequency + self.begin_time

    def time_interval(self) -> Tuple[float, float]:
        """ Return a sample containing the beginning and end time of the recording"""
        return self.begin_time, self.end_time

    def time_duration(self) -> float:
        """ Get the time duration expressed in seconds."""
        return self.num_samples() / self.frequency

    def channels(self) -> int:
        """ Returns the number of channels in the EEG recording which is assumed to be
        a 2d matrix as specified in the kaggle dataset.
        :return: the number of channels
        """
        return 0 if not self.data else np.size(self.data[0])

    def num_samples(self) -> int:
        """ Returns the number of samples in the EEG recording"""
        return int(np.size(self.data) / self.channels())

    def dealloc(self):
        """ Remove any reference internally to the data to decrease the reference count. """
        self.data = None
        del self.data
        # Ensure self.data is predictably None where no data is available.
        self.data = None

    @classmethod
    def from_path(cls, parent_directory: Union[str, Path], patient_id: int, night_id: int) -> Union[None, 'EEGRecording']:
        return EEGLoader.load(parent_directory, patient_id, night_id)

class EEGLoader:
    """ An helper loader class for the EEG recording data, which uses internally
    the default numpy loader for the npy data contained in the original Kaggle dataset at

    https://www.kaggle.com/datasets/yfrite/polysom?select=patients.csv
    """

    # As declared in , the sampling frequency is 400hz e.g. 2 times the cutoff frequency 100hz of gamma
    # waves
    DECLARED_FREQ = 200

    @staticmethod
    def load(parent_directory: Union[str, Path], patient_id: int, night_id: int) -> Union[EEGRecording, None]:
        """ Load an EEG associated with a given patient and a given night into an EEGRecording wrapper
        object.

        :param parent_directory: The directory containing all recordings for the patients, or simply containing
            the file.
        :param patient_id: The id of the patient as assigned in the csv file
        :param night_id: The id of the night as assigned in the csv file
        :return: an EEGRecording object as a rapper to the underlying data.
        """
        parent_directory = Path(parent_directory)
        file_name = EEGLoader._ids_to_file_name(patient_id, night_id)
        full_path = join(parent_directory, file_name)
        if not isfile(full_path):
            warn(f"Attempted to load data at {full_path} but it doesn't exist".format(full_path=full_path))
            return None

        # Use the default numpy loader
        data = np.load(full_path)
        eeg_data = EEGRecording(
            EEGLoader.DECLARED_FREQ,
            data=data,
            t0=0.0
        )
        eeg_data.give_ids(patient_id, night_id)
        return eeg_data

    @staticmethod
    def _ids_to_file_name(patient_id: int, night_id: int) -> str:
        """ Create the data path from the two ids, as specified in the kaggle dataset."""
        return f"User-{patient_id}-Night-{night_id}.npy"