import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Splits dataset into train and test subsets
from sklearn.model_selection import train_test_split

# Custom loader for the CSV dataset
from loader.patients import PatientsCSVLoader

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, cross_validate

# Path to the embedding matrix created by the FPCA pipeline
EMBEDDINGS_PATH = (
    r"C:\Users\tomma\.cache\kagglehub\datasets"
    r"\yfrite\polysom\versions\3\Embeddings"
    r"\embeddings.npy"
)

# Path to the patient metadata CSV
PATIENTS_CSV_PATH = (
    r"C:\Users\tomma\.cache\kagglehub\datasets"
    r"\yfrite\polysom\versions\3\patients.csv"
)


def build_samples():
    """
    Create the list of all (patient_id, night_id) pairs.
    Two recordings are removed because they are problematic
    """

    samples = []

    for patient_id in range(1, 40):
        for night_id in range(1, 3):
            samples.append((patient_id, night_id))

    # Remove problematic recordings
    samples.remove((8, 1))
    samples.remove((14, 2))

    return samples


def load_targets(samples):
    """
    Build the regression target vector y.
    For each recording: (patient_id, night_id) we extract NAp = number of apnea attacks
    Missing values are interpreted as 0 .
    """

    # Load patient dataframe
    df = PatientsCSVLoader.load_dataframe(PATIENTS_CSV_PATH)

    y = []

    for patient_id, night_id in samples:

        # Select the row corresponding to this recording
        row = df[
            (df["user_id"] == patient_id)
            &
            (df["night_id"] == night_id)
        ]

        # Extract apnea count
        attacks = row.iloc[0]["NAp"]

        # Handle missing values
        if pd.isna(attacks) or attacks == "" or attacks == "NaN":
            attacks = 0
        else:
            attacks = int(attacks)

        y.append(attacks)

    return np.array(y)


def main():

    # All the evaluated models
    models = {
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

    # Build sample list
    samples = build_samples()

    # Load embeddings
    # X.shape = (n_recordings, embedding_dimension)
    X = np.load(EMBEDDINGS_PATH)

    # Load regression targets
    # y.shape = (n_recordings,)
    y = load_targets(samples)

    plt.hist(y, bins=20)

    plt.title("Distribution of NAp")
    plt.xlabel("Number of apnea attacks")
    plt.ylabel("Count")

    plt.show()

    # Used to handle skewed values (log1p(x) = log(1 + x))
    y_model = np.log1p(y)

    # Train/test split (80% training 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_model,
        test_size=0.2,
        random_state=42
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }

    results = []

    for name, model in models.items():
        cv_results = cross_validate(
            model,
            X,
            y_model,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )

        mae_mean = -cv_results["test_MAE"].mean()
        mae_std = cv_results["test_MAE"].std()

        rmse_mean = -cv_results["test_RMSE"].mean()
        rmse_std = cv_results["test_RMSE"].std()

        r2_mean = cv_results["test_R2"].mean()
        r2_std = cv_results["test_R2"].std()

        train_r2_mean = cv_results["train_R2"].mean()

        results.append({
            "model": name,
            "CV_MAE_mean": mae_mean,
            "CV_MAE_std": mae_std,
            "CV_RMSE_mean": rmse_mean,
            "CV_RMSE_std": rmse_std,
            "CV_R2_mean": r2_mean,
            "CV_R2_std": r2_std,
            "Train_R2_mean": train_r2_mean
        })

        print(f"\n{name}")
        print(f"CV MAE:  {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"CV RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
        print(f"CV R²:   {r2_mean:.4f} ± {r2_std:.4f}")
        print(f"Train R²: {train_r2_mean:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("CV_MAE_mean")

    print("\nCross-validation model comparison:")
    print(results_df)

    top3_names = results_df.iloc[:3]["model"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, model_name in zip(axes, top3_names):
        model = models[model_name]

        # Train model
        model.fit(X_train, y_train)

        # Predict in log-space
        y_pred_log = model.predict(X_test)

        # Convert back to original apnea counts
        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test)

        # Scatter plot
        ax.scatter(y_test_real, y_pred_real)

        # Perfect prediction line
        min_v = min(y_test_real.min(), y_pred_real.min())
        max_v = max(y_test_real.max(), y_pred_real.max())

        ax.plot([min_v, max_v], [min_v, max_v])

        # Metrics (still computed in log-space)
        r2 = r2_score(y_test, y_pred_log)

        ax.set_title(f"{model_name}\nR² = {r2:.3f}")

        ax.set_xlabel("True NAp")
        ax.set_ylabel("Predicted NAp")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()