import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

def main():
    pass

if __name__ == "__main__":
    main()