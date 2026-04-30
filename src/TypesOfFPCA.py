import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)



class TypesOfFPCA:
    """
    This class implements different kind of FPCA:
    - Discretized FPCA
    - B-spline FPCA
    - Fourier FPCA
    - Monomial FPCA

    And it is able to plot them.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_optimal_n_components(fd, n_basis, threshold=0.95):
        """
        It finds the n component necessary to represent the threshold percentage of data.
        If threshold is 0.95, it returns the n sufficient to represent the 95% of the data.

        Args:
            fd (FDataGrid): It contains the functional data.
            n_basis (int): The number of basis functions to use.
            threshold (float): The percentage of the original data we want to represent (default is 0.95).
        Returns:
            int: The minimum number of components needed to represent the data for the given threshold.
        """

        # Executing a FPCA with the highest possible number of components
        max_comps = min(len(fd), n_basis) # cannot exceed the max length of fd
        fpca = FPCA(n_components=max_comps)
        fpca.fit(fd)

        # Computing cumulated variance
        # cumsum([0.7, 0.2, 0.05]) -> [0.7, 0.9, 0.95]
        cumulated_variance = np.cumsum(fpca.explained_variance_ratio_)

        # finding the first index that exceed the threshold
        n_optimal = np.argmax(cumulated_variance >= threshold) + 1

        print(f"Cumulated variance for component: {cumulated_variance}")
        print(f"To represent the {threshold * 100}% of the variance we need {n_optimal} components.")

        return n_optimal

    @staticmethod
    def draw_pca_graph(fpca, n_comps):
        """
        This method shows at video the fpca graph.
        Args:
            fpca (ndarray): functional pca we want to draw
            n_comps (int): The number of components to used to represent the fpca
        Returns:
            None
        """
        fpca.components_.plot()

        legend_names = [f"PC{i + 1} ({v * 100:.1f}%)" for i, v in enumerate(fpca.explained_variance_ratio_)]
        plt.legend(legend_names, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"First {n_comps} main functional components")
        plt.tight_layout()

        plt.show()

    @staticmethod
    def discretized_fpca(fd, n_basis, threshold = 0.95, toDraw=False):
        """
        Performs Discretized Functional Principal Component Analysis (FPCA).

        This method transforms functional data into a finite-dimensional space using
        basis expansion and identifies the principal modes of variation.

        Args:
            fd (FDataGrid): The functional data object containing the observed samples.
            n_basis (int): The number of basis functions (e.g., B-splines or Fourier) used to represent the functional data.
            threshold (float): The fraction of total variance that the selected components should
            explain. (default 0.95)
            toDraw (bool): boolean variable to set True if you want to draw the PCs. (default False)

        Returns:
            fd (FDataGrid): The functional data object containing the observed samples. Note: despite the method returns
            one of its input parameters unchanged, it has been done to follow the return structure of other fpca methods
            described in this class.
            fpca_discretized (ndarray): discretized fpca.
        """

        n_comps = TypesOfFPCA.get_optimal_n_components(fd, n_basis, threshold)

        fpca_discretized = FPCA(n_components=n_comps, # components shown at video, no larger than original data
                                centering=True, # Subtract the average curve of the data
                                regularization=None, # Penalizing outliers
                                components_basis=None, # Discretized FPCA
                                _weights=None # Giving more importance to some part of the wave despite others
                                )
        # fitting algorithm
        fpca_discretized.fit(fd)

        if toDraw:
            TypesOfFPCA.draw_pca_graph(fpca_discretized, n_comps)

        return fd, fpca_discretized

    @staticmethod
    def b_spline_fpca(fd, n_basis,threshold=0.95, toDraw=False):
        """
        Performs Functional Principal Component Analysis using a B-spline basis.

        This method projects discrete functional observations onto a B-spline
        basis to smooth the data before performing eigen-decomposition on the
        coefficient covariance matrix

        Args:
            fd (FDataGrid): The functional data object containing the observed samples.
            n_basis (int): The number of basis functions (e.g., B-splines or Fourier) used to represent the functional data.
            threshold (float): The fraction of total variance that the selected components should
            explain. (default 0.95)
            toDraw (bool): boolean variable to set True if you want to draw the PCs. (default False)

        Returns:
            basis_fd (FDataGrid): The functional data object containing the observed samples in B-Spline basis.
            fpca (ndarray): fpca using B-spline.
        """

        # Defining B-Spline
        basis = BSplineBasis(
            n_basis=n_basis,
            # order=2, # polynomial order: degree = order - 1
            # 4 -> cubic, 3 -> parabolic, 2 -> lines
            # domain_range=None, # If not specified, taken by the data
            # knots=None # positioning nodes more dense in the specified region, equidistant if not specified
        )
        # Smoothing
        basis_fd = fd.to_basis(basis)

        # showing original dataset
        #basis_fd.plot()
        #plt.title('BSpline')

        n_comps = TypesOfFPCA.get_optimal_n_components(basis_fd, n_basis, threshold=threshold)

        # computing FPCA
        fpca = FPCA(n_components=n_comps)
        fpca.fit(basis_fd)

        # plotting
        fpca.components_.plot()

        if toDraw:
            TypesOfFPCA.draw_pca_graph(fpca, n_comps)

        return basis_fd, fpca

    @staticmethod
    def fourier_fpca(fd, n_basis, threshold=0.95, toDraw=False):
        """
        Performs Functional Principal Component Analysis using a Fourier basis.

        This method is ideal for periodic functional data. It decomposes the
        curves into a series of sine and cosine functions to identify the
        dominant oscillatory patterns.

        Args:
            fd (FDataGrid): The functional data object containing the observed samples.
            n_basis (int): The number of basis functions (e.g., B-splines or Fourier) used to represent the functional data.
            threshold (float): The fraction of total variance that the selected components should
            explain. (default 0.95)
            toDraw (bool): boolean variable to set True if you want to draw the PCs. (default False)

        Returns:
            basis_fd (FDataGrid): The functional data object containing the observed samples in B-Spline basis.
            fpca (ndarray): fpca using Fourier.
        """
        fourier_basis = FourierBasis(n_basis=n_basis,
            # order=2, # polynomial order: degree = order - 1
            # 4 -> cubic, 3 -> parabolic, 2 -> lines
            # domain_range=None, # If not specified, taken by the data
            # knots=None # positioning nodes more dense in the specified region, equidistant if not specified
            )
        basis_fd = fd.to_basis(fourier_basis)
            # order=2, # polynomial order: degree = order - 1
            # 4 -> cubic, 3 -> parabolic, 2 -> lines
            # domain_range=None, # If not specified, taken by the data
            # knots=None # positioning nodes more dense in the specified region, equidistant if not specified

        n_comps = TypesOfFPCA.get_optimal_n_components(basis_fd, n_basis, threshold=threshold)

        fpca = FPCA(n_components=n_comps,  # components shown at video, no larger than original data
            centering=True,  # Subtract the average curve of the data
            #regularization=None,  # Penalizing outliers
            components_basis=fourier_basis,  # Fourier FPCA
            _weights=None  # Giving more importance to some part of the wave despite others
        )

        fpca.fit(basis_fd)

        if toDraw:
            TypesOfFPCA.draw_pca_graph(fpca, n_comps)

        return basis_fd, fpca

    @staticmethod
    def monomial_fpca(fd, n_basis, threshold=0.95, toDraw=False):
        """
        Performs Functional Principal Component Analysis using a Monomial basis.

        This method projects functional data onto a polynomial basis (1, t, t^2...).
        It is best suited for data exhibiting simple curvilinear trends rather than
        complex oscillations or local variations.

        Args:
            fd (FDataGrid): The functional data object containing the observed samples.
            n_basis (int): The number of basis functions (e.g., B-splines or Fourier) used to represent the functional data.
            threshold (float): The fraction of total variance that the selected components should
            explain. (default 0.95)
            toDraw (bool): boolean variable to set True if you want to draw the PCs. (default False)

        Returns:
            fd_basis (FDataGrid): The functional data object containing the observed samples in B-Spline basis.
            fpca (ndarray): fpca using monomial basis.
        """
        mono_basis = MonomialBasis(n_basis=n_basis)
        fd_basis = fd.to_basis(mono_basis)

        n_comps = TypesOfFPCA.get_optimal_n_components(fd_basis, n_basis, threshold)

        fpca = FPCA(n_components=n_comps, components_basis=mono_basis)
        fpca.fit(fd_basis)

        if toDraw:
            TypesOfFPCA.draw_pca_graph(fpca, n_comps)

        return fd_basis, fpca


    @staticmethod
    def plot_fpca(basis_fd, fpca, factor = 30):
        """
        FPCA plot visualization.
        Args:
            basis_fd (FDataGrid): The functional data object containing the observed samples.
            fpca (ndarray): fpca.
            factor (int): Multiple of the principal component curve to be added or subtracted. (default 30)

        Returns:
             None
        """
        n_components = len(fpca.components_)
        fig, axes = plt.subplots(nrows=n_components, ncols=1, figsize=(10, 5 * n_components))

        if n_components == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten()

        # Initializing plot
        plot = FPCAPlot(
            basis_fd.mean(),
            fpca.components_,
            factor=factor,
            axes=axes_list
        )
        plot.plot()

        for i, ax in enumerate(axes_list):
            lines = ax.get_lines()
            # In FPCAPlot:
            # lines[0] is the mean
            # Other lines are marker '+' and '-'

            color_media = lines[0].get_color()

            color_plus = lines[1].get_color() if len(lines) > 1 else 'gray'
            color_minus = lines[2].get_color() if len(lines) > 2 else 'gray'

            # Creating legends
            legend_elements = [
                Line2D([0], [0], color=color_media, lw=2, label='Mean ($\mu$)'),
                Line2D([0], [0], marker='+', color='none', markeredgecolor=color_plus,
                       markersize=10, label='Positive Variation (+)'),
                Line2D([0], [0], marker='_', color='none', markeredgecolor=color_minus,
                       markersize=10, label='Negative Variation (-)')
            ]

            ax.set_title(f"Principal component {i + 1}")
            ax.legend(handles=legend_elements, loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def main():
    pass


if __name__ == '__main__':
    main()