import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# user_id - Unique user id [Range 1-40]
# night_id - Night id of the row [Range 1-2]
# age - Age of the patient [Years]
# sex - sex of the patient [F: Female, M: Male]
# height - Height of the patient [Centimeters]
# weight - Weight of the patient [Kilograms]
# pulse - Pulse of the patient [Beats per minute]
# BPsys/BPdia - Systolic blood pressure / diastolic blood pressure of the patient [Millimeters of mercury/Millimeters of mercury]
# ODI - Oxygen desaturation index of the patient for the night. For some healthy patients will be none [Continuous range 0-…]
# NAp - Number of cases apnoe of the patient for the night. For healthy patients is none [Range 0-…]
# NHyp - Number of cases hyperapnoe of the patient for the night. For healthy patients is none [Range 0-…]
# AI - Apnoe index of the patient for the night. Calculated by the formula: NAp / TimeOfRecordInHours. For healthy patients is none [Continuous range 0-…]
# HI - Hyperapnoe index of the patient for the night. Calculated by the formula: NHyp / TimeOfRecordInHours. For healthy patients is none [Continuous range 0-…]
# AHI - Apnoe-hyperapnoe index of the patient for the night. Calculated by the formula: (NAp+NHyp) / TimeOfRecordInHours. For healthy patients is none [Continuous range 0-…]

class DrawingGraphs:
    """
        A utility class for loading, cleaning, and visualizing data.

        Drawing:
        - pie graph
        - correlation matrix
        - covariance matrix
        - numerical distribution
    """


    def __init__(self):
        raise SyntaxError("Cannot instantiate PatientsDSsetup, it is an utility static class!")

    @staticmethod
    def draw_correlation_matrix(df, num_vars):
        """
        Method to draw and compute the correlation matrix
        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            num_vars (list of str): A list of names of the numerical variables to analyze.

        Returns:
            pd.DataFrame: A square dataframe representing the correlation matrix
                of the selected variables.
        """
        df[num_vars] = df[num_vars].apply(pd.to_numeric, errors='coerce')
        corr_mat = df[num_vars].corr()

        plt.figure(figsize=(12, 9))
        sns.heatmap(corr_mat, cmap="Blues", annot=True)
        plt.title('Correlation Matrix Heatmap')
        plt.show()

        return corr_mat

    @staticmethod
    def draw_covariance_matrix(df, num_vars):
        """
        Method to draw and compute the covariance matrix
        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            num_vars (list of str): A list of names of the numerical variables to analyze.

        Returns:
             pd.DataFrame: A square dataframe representing the covariance matrix
                of the selected variables.
        """
        # Covariance matrix
        # Calculate the covariance matrix
        cov_mat = df[num_vars].cov()
        ## Plot the covariance matrix as a heatmap
        plt.figure(figsize=(12, 9))
        sns.heatmap(cov_mat, annot=True, cmap='Blues')

        plt.title('Covariance Matrix Heatmap')
        plt.show()

        return cov_mat

    @staticmethod
    def __func_label(pct, all_vals):
        """
        Private method, computing label for graph pie: counts + percentage.
        Note: the value is divided by two, due to the repetition of patients (2 nights * 20 patients)
        Args:
            pct (float): The percentage value of the current pie slice,
                provided automatically by matplotlib.
            all_vals (pd.Series or list): The absolute counts used to generate
                the pie chart, used to calculate the absolute value from the percentage.

        Returns:
              str: A formatted string containing the absolute count and the percentage.
        """
        absolute = int(round(pct / 100. * sum(all_vals))) / 2
        return f"{absolute}\n({pct:.1f}%)"

    @staticmethod
    def draw_pie_graph(df, var, colors, title, slice = None, slice_value=None):
        """
        Method to draw a pie graph
        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            var (str): The column name to be visualized (categorical variable).
            colors (list of str): A list of hexadecimal color codes.
                Note: The number of colors should match the number of unique categories in 'var'.
            title (str): The main title of the plot.
            slice (str, optional): The column name used to filter the dataset (e.g., 'sex').
                Defaults to None, meaning the entire population is included.
            slice_value (str, optional): The specific category within the 'slice' column
                to be analyzed (e.g., 'M'). Defaults to None.
                Only effective if 'slice' is provided.

        Returns:
            None: The method displays the plot using plt.show().
        """
        if slice and slice_value:
            var_counts = df[df[slice] == slice_value][var].value_counts()
        else:
            var_counts = df[var].value_counts()

        plt.pie(var_counts,
                autopct=lambda pct: DrawingGraphs.__func_label(pct, var_counts),
                labels=var_counts.index,
                startangle=90,
                colors=colors)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_numerical_distribution(df, num_vars, slice):
        """
        Method to plot the distribution of numerical variables
        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            num_vars (list of str): A list of names of the numerical variables to analyze.
            slice (str): The column name used to filter the dataset (e.g., 'sex').

        Returns:
              None: The method displays the plot using plt.show().
        """

        for col in num_vars:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_long = df.melt(id_vars=[slice],
                                    value_vars=num_vars,
                                    var_name='parameter',
                                    value_name='value'
                                    )

        # Plotting with displot
        g = sns.displot(
            data=df_long,
            x="value",
            hue=slice,
            col="parameter",
            kind="kde",
            col_wrap=3,  # 3 plots per row
            height=3,  # Height of each plot
            aspect=1.2,  # Width/Height ratio
            facet_kws={'sharex': False, 'sharey': False},
            fill=True,
            palette='muted'  # Easier on the eyes
        )

        # Use col_template to explicitly tell Seaborn how to format titles
        g.set_titles(col_template="{col_name}", pad=0)

        # Adjust the space between the subplots manually to prevent overlaps
        g.fig.subplots_adjust(hspace=0.4, wspace=0.2)

        g.set_axis_labels("", "Density")

        # Apply tight_layout to the figure object specifically
        g.fig.tight_layout()

        plt.show()

    @staticmethod
    def plot_PSD(fs, data):
        """
        This method plots the PSD (Power Spectral Density) of the data.
        PSD analysis is a frequency-domain technique used to identify how much power (or signal intensity)
        is distributed across different frequencies in a signal. It is the industry standard for analyzing random, complex vibrations or signals that
        persist over time. PSD helps determine the dominant frequencies in data, often used to predict, control, or analyze dynamic
        responses and vibrations

        Args:
            fs (float): The sample frequency.
            data (list of data): A list of numerical values to be plotted.

        Returns:
              None: The method displays the plot using plt.show().
        """
        n_samples = len(data)
        # x axes, preparating the time
        total_time = n_samples / fs
        ax = np.linspace(0, total_time, n_samples)

        plt.figure(figsize=(12, 8))

        # different bands for brain waves
        bands = [
            {'name': 'Delta', 'limits': (0.5, 4), 'color': 'gray'},
            {'name': 'Theta', 'limits': (4, 8), 'color': 'blue'},
            {'name': 'Alpha', 'limits': (8, 13), 'color': 'green'},
            {'name': 'Beta', 'limits': (13, 30), 'color': 'orange'},
            {'name': 'Gamma', 'limits': (30, 45), 'color': 'red'},
            {'name': 'Gamma+', 'limits': (45, 60), 'color': 'cyan'} #these waves are not actually relevant
        ]
        # figure 1: amplitude of the waves in microVolt in the time domain
        plt.subplot(211)
        plt.plot(ax[::100], data[::100], linewidth=0.5)
        plt.title(f"Time Signal (Duration: {total_time:.2f} s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude [\u03bcV]")

        # figure 2: Power analysis on a logarithmic scale [uV^2/Hz => dB/Hz]
        plt.subplot(212)
        plt.psd(data, NFFT=2048, Fs=fs, color='black', zorder=5)
        for b in bands:
            plt.axvspan(b['limits'][0], b['limits'][1],
                        color=b['color'], alpha=0.2, label=b['name'])
            # name of the band
            plt.text((b['limits'][0] + b['limits'][1]) / 2, -20, b['name'],
                     horizontalalignment='center', fontweight='bold', color=b['color'])

        plt.title("Spectral Analysis (Frequency Bands)")
        plt.xlim(0, 60) # limit on x axis
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_fourier_transform(fs, signal):
        """
            This method plots and computes the Fourier Transform of a given signal. the PSD (Power Spectral Density) of the data.

            Args:
                fs (float): The sample frequency.
                signal (list of data): A list of numerical values to be plotted.

            Returns:
                a: frequency (np.ndarray): Array of frequencies (X axes), from 0 to fs/2.

                transform (np.ndarray): Power density values (Y axes) in µV²/Hz.
        """
        # preparing the x axes (time)
        n_samples = len(signal)

        duration = n_samples / fs

        t = np.linspace(0, duration, n_samples)

        # Applying FFT
        frequency = np.fft.fftfreq(len(t), 1 / fs)
        transform = np.fft.fft(signal)

        # Just positive half (fft is specular)
        n = len(t) // 2

        # bandwidth of brain waves
        band_eeg = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 45)
        }


        f_plot = frequency[:n]
        amplitude_plot = np.abs(transform[:n])

        # Plotting
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(t, signal)
        plt.title("Signal in time domain (Raw EEG)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [\u03bcV]")

        plt.subplot(1, 2, 2)

        plt.plot(f_plot, amplitude_plot, color='black', lw=1)

        colors = ['gray', 'blue', 'green', 'red', 'purple']
        for i, (nome, (f_min, f_max)) in enumerate(band_eeg.items()):
            plt.axvspan(f_min, f_max, color=colors[i], alpha=0.2, label=nome)
        plt.title("Frequency Spectrum (Fourier)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r'$\mu V^2/Hz$')
        plt.legend()
        plt.xlim(0, 50)
        plt.subplots_adjust(wspace=0.3)
        plt.show()

        return frequency, transform

def main():
    pass

if __name__ == "__main__":
    main()