import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader.patients import PatientsCSVLoader

from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    r2_score
)

from src.PatientsDSsetup import PatientsDSsetup


AHI_THRESHOLD = 5.0


def build_samples():
    samples = []

    for patient_id in range(1, 40):
        for night_id in range(1, 3):
            samples.append((patient_id, night_id))

    samples.remove((8, 1))
    samples.remove((14, 2))

    groups = np.array([pid for pid, _ in samples])

    return samples, groups


def parse_numeric(value):
    if pd.isna(value):
        return np.nan

    value = str(value).replace(",", ".").strip()

    if value == "" or value.lower() == "nan":
        return np.nan

    return float(value)


def parse_ahi(value):
    """
    In this dataset, missing/NaN AHI means healthy patient,
    therefore it is converted to 0.0.
    """

    if pd.isna(value):
        return 0.0

    value = str(value).replace(",", ".").strip()

    if value == "" or value.lower() == "nan":
        return 0.0

    return float(value)


def parse_blood_pressure(value):
    if pd.isna(value):
        return np.nan, np.nan

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return np.nan, np.nan

    parts = value.split("/")

    if len(parts) != 2:
        return np.nan, np.nan

    return parse_numeric(parts[0]), parse_numeric(parts[1])


def load_dataset(samples):
    df = PatientsDSsetup.load_dataframe()
    df = PatientsDSsetup.add_BMI(df)
    #df = PatientsCSVLoader.load_dataframe(PATIENTS_CSV_PATH)

    X = []
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

        row = row.iloc[0]

        sex = 1.0 if row["sex"] == "M" else 0.0

        bp_sys, bp_dia = parse_blood_pressure(row["BPsys/BPdia"])

        features = [
            parse_numeric(row["age"]),
            sex,
            parse_numeric(row["height"]),
            parse_numeric(row["weight"]),
            parse_numeric(row["pulse"]),
            bp_sys,
            bp_dia,
        ]

        target = parse_ahi(row["AHI"])

        X.append(features)
        y.append(target)

    return np.array(X, dtype=float), np.array(y, dtype=float)


def build_models():
    return {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),

        "Lasso": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.05, max_iter=10000))
        ]),

        "ElasticNet": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                random_state=42
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42
            ))
        ]),

        "SVR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=1.0,
                gamma="scale"
            ))
        ])
    }


def evaluate_for_table(model, X, y_log, cv, groups ,threshold=AHI_THRESHOLD):
    mse_scores = []
    rmse_scores = []
    acc_scores = []
    prec_scores = []
    rec_scores = []
    mae_scores = []
    r2_scores = []

    for train_idx, test_idx in cv.split(X, y_log, groups=groups):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train_log = y_log[train_idx]
        y_test_log = y_log[test_idx]

        model.fit(X_train, y_train_log)

        y_pred_log = model.predict(X_test)

        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test_log)

        y_pred_real = np.maximum(y_pred_real, 0.0)

        mse = mean_squared_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mse)

        y_true_class = (y_test_real >= threshold).astype(int)
        y_pred_class = (y_pred_real >= threshold).astype(int)

        acc = accuracy_score(y_true_class, y_pred_class)
        prec = precision_score(y_true_class, y_pred_class, zero_division=0)
        rec = recall_score(y_true_class, y_pred_class, zero_division=0)
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        mae_scores.append(mae)
        r2_scores.append(r2)

    return {
        "MSE": np.mean(mse_scores),
        "RMSE": np.mean(rmse_scores),
        "Acc.": np.mean(acc_scores),
        "Prec.": np.mean(prec_scores),
        "Rec.": np.mean(rec_scores),
        "MAE": np.mean(mae_scores),
        "R²": np.mean(r2_scores)
    }


def compare_models(models, X, y_log, groups):
    cv = GroupShuffleSplit(
        n_splits=100,
        test_size=0.2,
        random_state=42
    )

    results = []

    for name, model in models.items():
        metrics = evaluate_for_table(
            model,
            X,
            y_log,
            cv,
            groups,
            threshold=AHI_THRESHOLD
        )

        row = {
            "Model": name,
            **metrics
        }

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")

    print("\nResults for slide table:")
    print(results_df.round(3))

    return results_df


def plot_top_models(models, results_df, X, y_log):
    top3_names = results_df.iloc[:3]["Model"].values

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=0.2,
        random_state=42
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, model_name in zip(axes, top3_names):
        model = models[model_name]

        model.fit(X_train, y_train_log)

        y_pred_log = model.predict(X_test)

        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test_log)

        y_pred_real = np.maximum(y_pred_real, 0.0)

        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        r2 = r2_score(y_test_real, y_pred_real)

        ax.scatter(y_test_real, y_pred_real)

        min_v = min(y_test_real.min(), y_pred_real.min())
        max_v = max(y_test_real.max(), y_pred_real.max())

        ax.plot([min_v, max_v], [min_v, max_v])

        ax.set_title(
            f"{model_name}\n"
            f"R² = {r2:.3f}\n"
            f"MAE = {mae:.2f}, RMSE = {rmse:.2f}"
        )

        ax.set_xlabel("True AHI")
        ax.set_ylabel("Predicted AHI")

    plt.tight_layout()
    plt.show()


def main():
    samples, groups = build_samples()

    X, y = load_dataset(samples)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    y_log = np.log1p(y)

    models = build_models()

    results_df = compare_models(models, X, y_log, groups)

    print("\nCopy these values into the slide table:")
    print(results_df.round(3).to_string(index=False))

    plot_top_models(models, results_df, X, y_log)


if __name__ == "__main__":
    main()