from scipy.signal import butter, filtfilt, iirnotch

class Filter:
    def __init__(self, fs, ftype, cutoff = None , order = 2, lowcut = None, highcut = None, quality = None, notch_frequency = None):
        self.ftype = ftype

        # check on pass low, high filter
        if ftype in ['low', 'high']:
            if cutoff is None:
                raise ValueError(f"For the filter '{ftype}', the parameter 'cutoff' is mandatory.")
            self.cutoff = cutoff

        # Check on Pass Band
        elif ftype == 'band':
            if lowcut is None or highcut is None:
                raise ValueError("For the band filter, 'lowcut' and 'highcut' are mandatory.")
            if lowcut >= highcut:
                raise ValueError(f"Logic error: lowcut ({lowcut}) must be lower than highcut ({highcut}).")
            self.lowcut = lowcut
            self.highcut = highcut

        # Check on notch filter
        elif ftype == 'notch':
            if notch_frequency is None or quality is None:
                raise ValueError("For the 'notch' filter, 'notch_frequency' and 'quality' are mandatory.")
            self.notch_freq = notch_frequency
            self.quality = quality

        # Unknown type
        else:
            raise ValueError(f"The filter '{ftype}' is not known or implemented. Use 'low', 'high', 'band' or 'notch'.")

        self.order = order
        self.b, self.a = self.butter_filter()

    def butter_filter(self):
        if(self.ftype == 'band'):
            return butter(self.order, [self.lowcut, self.highcut], fs=self.fs, btype=self.ftype, analog=False)
        elif(self.ftype != 'notch'):
            return butter(self.order, self.cutoff, fs=self.fs, btype=self.ftype, analog=False)

    def apply_filter(self, data):
        y = filtfilt(self.b, self.a, data)
        return y

class LowPassFilter:
    def __init__(self, cutoff, fs, order):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.ny = 0.5 * fs

        self.b, self.a = self.butter_lowpass()


    def butter_lowpass(self):
        return butter(self.order, self.cutoff, fs=self.fs, btype='low', analog=False)

    def butter_lowpass_filter(self, data):
        return filtfilt(self.b, self.a, data)

class HighPassFilter:
    def __init__(self, cutoff, fs, order):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.ny = 0.5 * fs

        self.b, self.a = self.butter_highpass()


    def butter_highpass(self):
        return butter(self.order, self.cutoff, fs=self.fs, btype='high', analog=False)

    def butter_highpass_filter(self, data):
        y = filtfilt(self.b, self.a, data)
        return y

class BandPassFilter:
    def __init__(self, lowcut, highcut, fs, order):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.ny = 0.5 * fs

        self.b, self.a = self.butter_bandpass()

    def butter_bandpass(self):
        return butter(self.order, [self.lowcut, self.highcut], fs=self.fs, btype='band', analog=False)

    def butter_bandpass_filter(self, data):
        y = filtfilt(self.b, self.a, data)
        return y

class NotchFilter:
    def __init__(self, notch_freq, fs, quality):
        self.notch_freq = notch_freq
        self.fs = fs
        self.quality = quality
        self.ny = 0.5 * fs

        self.b, self.a = iirnotch(self.notch_freq, self.quality, self.fs)


    def butter_notch_filter(self, data):
        y = filtfilt(self.b, self.a, data)
        return y