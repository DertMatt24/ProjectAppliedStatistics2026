import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader.patients import PatientsCSVLoader

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


EMBEDDINGS_PATH = (
    r"C:\Users\tomma\.cache\kagglehub\datasets"
    r"\yfrite\polysom\versions\3\Embeddings"
    r"\embeddings.npy"
)

PATIENTS_CSV_PATH = (
    r"C:\Users\tomma\.cache\kagglehub\datasets"
    r"\yfrite\polysom\versions\3\patients.csv"
)


def build_samples():
    samples = []

    for patient_id in range(1, 40):
        for night_id in range(1, 3):
            samples.append((patient_id, night_id))

    samples.remove((8, 1))
    samples.remove((14, 2))

    return samples


def parse_ahi(value):
    """
    Convert AHI values to float.

    In this dataset, missing/NaN AHI means healthy patient,
    so it is converted to 0.0.
    """

    if pd.isna(value):
        return 0.0

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return 0.0

    value = value.replace(",", ".")

    return float(value)


def load_targets(samples):
    df = PatientsCSVLoader.load_dataframe(PATIENTS_CSV_PATH)

    y = []

    for patient_id, night_id in samples:
        row = df[
            (df["user_id"] == patient_id)
            & (df["night_id"] == night_id)
        ]

        if row.empty:
            raise ValueError(
                f"No row found for patient_id={patient_id}, night_id={night_id}"
            )

        ahi = parse_ahi(row.iloc[0]["AHI"])
        y.append(ahi)

    return np.array(y, dtype=float)


def build_models():
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),

        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, max_iter=10000))
        ]),

        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }


def evaluate_real_scale(model, X, y_log, cv):
    """
    Train on log1p(AHI), but evaluate predictions after converting
    back to the original AHI scale.
    """

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_log, y_test_log = y_log[train_idx], y_log[test_idx]

        model.fit(X_train, y_train_log)

        y_pred_log = model.predict(X_test)

        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test_log)

        y_pred_real = np.maximum(y_pred_real, 0.0)

        mae_scores.append(mean_absolute_error(y_test_real, y_pred_real))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
        r2_scores.append(r2_score(y_test_real, y_pred_real))

    return {
        "MAE_real_mean": np.mean(mae_scores),
        "MAE_real_std": np.std(mae_scores),
        "RMSE_real_mean": np.mean(rmse_scores),
        "RMSE_real_std": np.std(rmse_scores),
        "R2_real_mean": np.mean(r2_scores),
        "R2_real_std": np.std(r2_scores),
    }


def compare_models(models, X, y_log):
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=20,
        random_state=42
    )

    scoring = {
        "MAE_log": "neg_mean_absolute_error",
        "RMSE_log": "neg_root_mean_squared_error",
        "R2_log": "r2"
    }

    results = []

    for name, model in models.items():
        cv_log = cross_validate(
            model,
            X,
            y_log,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )

        real_metrics = evaluate_real_scale(model, X, y_log, cv)

        row = {
            "model": name,

            "CV_MAE_log_mean": -cv_log["test_MAE_log"].mean(),
            "CV_MAE_log_std": cv_log["test_MAE_log"].std(),

            "CV_RMSE_log_mean": -cv_log["test_RMSE_log"].mean(),
            "CV_RMSE_log_std": cv_log["test_RMSE_log"].std(),

            "CV_R2_log_mean": cv_log["test_R2_log"].mean(),
            "CV_R2_log_std": cv_log["test_R2_log"].std(),

            "Train_R2_log_mean": cv_log["train_R2_log"].mean(),

            **real_metrics
        }

        results.append(row)

        print(f"\n{name}")
        print(f"Log-space CV R²: {row['CV_R2_log_mean']:.4f} ± {row['CV_R2_log_std']:.4f}")
        print(f"Real-scale MAE:  {row['MAE_real_mean']:.4f} ± {row['MAE_real_std']:.4f}")
        print(f"Real-scale RMSE: {row['RMSE_real_mean']:.4f} ± {row['RMSE_real_std']:.4f}")
        print(f"Real-scale R²:   {row['R2_real_mean']:.4f} ± {row['R2_real_std']:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE_real_mean")

    return results_df


def plot_target_distribution(y):
    plt.figure(figsize=(7, 4))
    plt.hist(y, bins=20)
    plt.title("Distribution of AHI")
    plt.xlabel("AHI")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_top_models(models, results_df, X_train, X_test, y_train_log, y_test_log):
    top3_names = results_df.iloc[:3]["model"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, model_name in zip(axes, top3_names):
        model = models[model_name]

        model.fit(X_train, y_train_log)

        y_pred_log = model.predict(X_test)

        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test_log)

        y_pred_real = np.maximum(y_pred_real, 0.0)

        r2_real = r2_score(y_test_real, y_pred_real)
        mae_real = mean_absolute_error(y_test_real, y_pred_real)
        rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

        ax.scatter(y_test_real, y_pred_real)

        min_v = min(y_test_real.min(), y_pred_real.min())
        max_v = max(y_test_real.max(), y_pred_real.max())

        ax.plot([min_v, max_v], [min_v, max_v])

        ax.set_title(
            f"{model_name}\n"
            f"R² real = {r2_real:.3f}\n"
            f"MAE = {mae_real:.2f}, RMSE = {rmse_real:.2f}"
        )

        ax.set_xlabel("True AHI")
        ax.set_ylabel("Predicted AHI")

    plt.tight_layout()
    plt.show()


def plot_residuals(best_model, X_train, X_test, y_train_log, y_test_log):
    best_model.fit(X_train, y_train_log)

    y_pred_log = best_model.predict(X_test)

    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test_log)

    y_pred_real = np.maximum(y_pred_real, 0.0)

    residuals = y_test_real - y_pred_real

    plt.figure(figsize=(7, 4))
    plt.scatter(y_pred_real, residuals)
    plt.axhline(0, linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted AHI")
    plt.ylabel("Residual = True AHI - Predicted AHI")
    plt.tight_layout()
    plt.show()


def main():
    samples = build_samples()

    X = np.load(EMBEDDINGS_PATH)
    y = load_targets(samples)

    if len(X) != len(y):
        raise ValueError(
            f"X and y have different lengths: X={len(X)}, y={len(y)}"
        )

    plot_target_distribution(y)

    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=0.2,
        random_state=42
    )

    models = build_models()

    results_df = compare_models(models, X, y_log)

    print("\nModel comparison sorted by real-scale MAE:")
    print(results_df)

    plot_top_models(
        models,
        results_df,
        X_train,
        X_test,
        y_train_log,
        y_test_log
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = models[best_model_name]

    print(f"\nBest model according to real-scale MAE: {best_model_name}")

    plot_residuals(
        best_model,
        X_train,
        X_test,
        y_train_log,
        y_test_log
    )


if __name__ == "__main__":
    main()