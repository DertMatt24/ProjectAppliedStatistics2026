import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV, RFE
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import pandas as pd

class FeatureSelection:
    """
    Static class, implementing various algorithm for feature selection
    """
    def __init__(self):
        raise SyntaxError("Cannot instantiate FeatureSelection, it is static class!")

    @staticmethod
    def fs_score(model, X, y, selected_features, cv=5):
        """
        Compares model performance using all features versus a selected subset.

        :param model: The machine learning model to evaluate (e.g., Random Forest or Linear Regression).
        :param X: The full feature matrix (original data).
        :param y: The target vector (e.g., AHI scores).
        :param selected_features: List of column names chosen by your selection algorithm.
        :param cv: Number of cross-validation folds. Defaults to 5.
        """
        # Score with EVERYTHING
        score_all = cross_val_score(model, X, y, cv=cv, scoring='r2').mean()

        # Score with ONLY SELECTED
        score_sel = cross_val_score(model, X[selected_features], y, cv=cv, scoring='r2').mean()

        print(f"R2 with All Features: {score_all:.4f}")
        print(f"R2 with Selected Features: {score_sel:.4f}")
        print(f"Difference: {score_sel - score_all:.4f}")

    @staticmethod
    def forward_selection(model, X, y, cv,n_features_to_select):
        """
        Performs Forward Feature Selection to identify the most predictive variables.

        :param model: The machine learning algorithm (e.g., RandomForest or LinearRegression).
        :param X: The predictor variables.
        :param y: The target variable.
        :param cv: Number of cross-validation folds (e.g., 5).
        :param n_features_to_select: How many features you want to end up with.
        :return: A list of selected feature names and the fitted selector object.
        """
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select=n_features_to_select,
            direction='forward',
            scoring='r2',
            cv=cv
        )

        sfs.fit(X, y)

        selected_features = X.columns[sfs.get_support()]
        return selected_features, sfs

    @staticmethod
    def be_score(X,y, model_sel):
        """
        Evaluates the success of Backward Elimination by comparing the
        Adjusted R-Squared of the full model versus the reduced model.

        :param X: The original full feature matrix (before elimination).
        :param y: The target variable (e.g., AHI scores).
        :param model_sel: The final fitted OLS model returned by backward_elimination.
        """
        X_all = sm.add_constant(X)
        model_all = sm.OLS(y, X_all).fit()

        print(f"Original Adj. R-Squared: {model_all.rsquared_adj:.4f}")
        print(f"Final Adj. R-Squared: {model_sel.rsquared_adj:.4f}")

    @staticmethod
    def backward_elimination(X, y, significance_level=0.05):
        """
            Performs backward elimination based on p-values to find a statistically
            significant subset of features for a Linear Regression model.

            :param X: Predictor variables (DataFrame).
            :param y: Target variable (Series/Array).
            :param significance_level: The alpha threshold (usually 0.05).
                                       Variables with p-values above this are removed.
            :return: A tuple containing (list of selected features, the final fitted model).
        """
        X = sm.add_constant(X)
        features = list(X.columns)

        while len(features) > 0:
            model = sm.OLS(y, X[features]).fit()
            # Filter p-values to ignore the constant
            p_values = model.pvalues.drop('const', errors='ignore')

            if p_values.empty: break  # Only const remains

            max_p_value = p_values.max()
            if max_p_value > significance_level:
                excluded_feature = p_values.idxmax()
                features.remove(excluded_feature)
                print(f"Removed: {excluded_feature} (p-value: {max_p_value:.4f})")
            else:
                break

        features.remove('const')
        model_sel = sm.OLS(y, X[features]).fit()

        return features, model_sel

    @staticmethod
    def recursive_feature_elimination_kf(model,cv):
        """
        Performs Recursive Feature Elimination with Cross-Validation (RFECV).
        This automatically finds the optimal number of features by evaluating
        model performance on different subsets of the data.

        :param model: The machine learning estimator (e.g., RandomForest or LinearRegression).
        :param cv: The cross-validation strategy (e.g., 5 or 10 folds).
        :return: The fitted RFECV selector object containing the results.
        """
        selector = RFECV(estimator=model,
        step=1,
        cv=cv,
        scoring='r2',
        min_features_to_select=2)

        return selector


    @staticmethod
    def recursive_feature_elimination(model, X, y, n_to_select):
        """
        Manually executes Recursive Feature Elimination to reach a specific
        number of desired features.

        :param model: The supervised learning estimator (must have a 'coef_' or 'feature_importances_' attribute).
        :param X: The predictor variables (DataFrame).
        :param y: The target variable (e.g., AHI scores).
        :param n_to_select: The exact number of features to keep.
        :return: The fitted RFE selector object.
        """
        selector = RFE(estimator=model, n_features_to_select=n_to_select)
        selector.fit(X,y)

        # See which features survived
        selected_mask = selector.support_
        chosen_columns = X.columns[selected_mask]

        print(f"The best {len(chosen_columns)} features are: {chosen_columns.tolist()}")

        # See the ranking of the 'losers'
        #ranking = selector.ranking_
        # Features with ranking 1 are the ones selected.
        # Ranking 2 was the last one eliminated, and so on.
        #print(ranking)

        return selector

    @staticmethod
    def computing_mi_scores(X, y):
        """
        Computes and visualizes Mutual Information scores between features and the target.
        MI captures non-linear dependencies that standard correlation might miss.

        :param X: The predictor variables (DataFrame).
        :param y: The target variable (e.g., AHI scores).
        :return: The raw numpy array of MI scores.
        """
        mi_scores = mutual_info_regression(X, y)
        mi_series = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_series = mi_series.sort_values(ascending=False)

        print(mi_series)
        FeatureSelection.plot_mi_scores(mi_series, X)

        return mi_scores

    @staticmethod
    def plot_mi_scores(scores, X):
        """
        Creates a horizontal bar chart of Mutual Information scores for visual comparison.

        :param scores: The raw MI scores (array or series) calculated from the data.
        :param X: The feature DataFrame, used to extract column names for the labels.
        """
        mi_series = pd.Series(scores, index=X.columns)
        mi_series = mi_series.sort_values(ascending=True)

        width = np.arange(len(mi_series))
        ticks = list(mi_series.index)
        plt.barh(width, mi_series)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")
        plt.xlabel("Predictive Power")

        plt.show()

def main():
    pass


if __name__ == '__main__':
    main()