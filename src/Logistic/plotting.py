import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_nonzero_odds_ratios(result, feature_names=None, eps=1e-8):
    """
    Forest plot of non-zero odds ratios from a statsmodels MNLogit result.

    Works especially well after:
        result = model.fit_regularized(method="l1", alpha=...)

    Parameters
    ----------
    result:
        Fitted statsmodels MNLogit result.
    feature_names:
        Optional list of names for predictors, excluding the constant.
        If None, generic x1, x2, ... names are used.
    eps:
        Coefficients with abs(coef) <= eps are treated as zero.
    """

    params = result.params.copy()

    # Rename rows if feature names are provided
    if feature_names is not None:
        new_index = ["const"] + list(feature_names)

        if len(new_index) == len(params.index):
            params.index = new_index
        else:
            print(
                "Warning: feature_names length does not match params. "
                "Using default names."
            )

    # Confidence intervals may contain NaN for coefficients set to zero by L1.
    ci = result.conf_int()

    summaries = {}

    for cls in params.columns:
        coef = params[cls]

        # Keep only non-zero coefficients and remove intercept
        mask = (coef.abs() > eps)
        mask.loc["const"] = False if "const" in mask.index else False

        coef_nonzero = coef[mask]

        if coef_nonzero.empty:
            continue

        or_values = np.exp(coef_nonzero)

        # Get CI only for selected coefficients
        conf = ci.loc[cls].loc[coef_nonzero.index]
        conf_or = np.exp(conf)

        summaries[cls] = pd.DataFrame({
            "Coefficient": coef_nonzero,
            "Odds Ratio": or_values,
            "2.5% CI": conf_or["2.5%"],
            "97.5% CI": conf_or["97.5%"]
        })

    if not summaries:
        print("No non-zero coefficients found.")
        return

    n_classes = len(summaries)

    fig, axes = plt.subplots(
        1,
        n_classes,
        figsize=(7 * n_classes, 6),
        sharey=False
    )

    if n_classes == 1:
        axes = [axes]

    for ax, (cls, summary) in zip(axes, summaries.items()):
        y_pos = np.arange(len(summary))

        ax.errorbar(
            summary["Odds Ratio"],
            y_pos,
            xerr=[
                summary["Odds Ratio"] - summary["2.5% CI"],
                summary["97.5% CI"] - summary["Odds Ratio"]
            ],
            fmt="o",
            ecolor="lightgray",
            elinewidth=3,
            capsize=4
        )

        ax.axvline(1, color="black", linestyle="--")
        ax.set_xscale("log")
        ax.set_title(f"Class: {cls}")
        ax.set_xlabel("Odds Ratio, log scale")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary.index)

    plt.suptitle("Non-zero odds ratios with 95% confidence intervals")
    plt.tight_layout()
    plt.show()