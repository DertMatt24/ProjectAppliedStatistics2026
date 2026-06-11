import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader.patients import PatientsCSVLoader

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score
)


class MultimodalAHIRegressor:
    """
    Regression model for predicting AHI.

    Modes:
        clinical    -> only clinical variables
        wavelet     -> only wavelet embeddings
        multimodal  -> clinical + wavelet embeddings

    Regressors:
        ridge
        random_forest
    """

    CLASS_NAMES = ["healthy", "moderate", "severe"]

    def __init__(
        self,
        patients_csv_path,
        wavelet_embeddings_path,
        completed_samples_path,
        mode="multimodal",
        regressor_type="ridge",
        n_splits=5,
        alpha=10.0,
        n_estimators=500,
        max_depth=None,
        random_state=42
    ):
        self.patients_csv_path = patients_csv_path
        self.wavelet_embeddings_path = wavelet_embeddings_path
        self.completed_samples_path = completed_samples_path

        self.mode = mode
        self.regressor_type = regressor_type
        self.n_splits = n_splits
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.df = None
        self.X = None
        self.y = None
        self.groups = None
        self.model = None

    @staticmethod
    def _parse_numeric(series):
        return pd.to_numeric(series, errors="coerce")

    @staticmethod
    def ahi_to_class(ahi):
        if ahi < 5:
            return 0
        elif ahi < 30:
            return 1
        else:
            return 2

    def _load_and_align_data(self):
        X_wavelet = np.load(self.wavelet_embeddings_path)
        completed_samples = np.load(self.completed_samples_path)

        samples_df = pd.DataFrame(
            completed_samples,
            columns=["user_id", "night_id"]
        )

        samples_df["user_id"] = pd.to_numeric(samples_df["user_id"], errors="coerce")
        samples_df["night_id"] = pd.to_numeric(samples_df["night_id"], errors="coerce")

        df = PatientsCSVLoader.load_dataframe(self.patients_csv_path)

        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
        df["night_id"] = pd.to_numeric(df["night_id"], errors="coerce")

        df = samples_df.merge(
            df,
            on=["user_id", "night_id"],
            how="left"
        )

        if len(df) != len(X_wavelet):
            raise ValueError(
                f"Alignment error: df has {len(df)} rows, "
                f"X_wavelet has {len(X_wavelet)} rows."
            )

        alignment_ok = (
            df[["user_id", "night_id"]].to_numpy()
            == completed_samples
        ).all()

        if not alignment_ok:
            raise ValueError("Row alignment between df and X_wavelet failed.")

        self.df = df

        return df, X_wavelet

    def _build_target(self, df):
        df["AHI"] = pd.to_numeric(
            df["AHI"],
            errors="coerce"
        )

        return df["AHI"].to_numpy(dtype=float)

    def _build_clinical_features(self, df):
        X_patients = df[[
            "user_id",
            "night_id",
            "age",
            "sex",
            "height",
            "weight",
            "pulse"
        ]].copy()

        X_patients["age"] = self._parse_numeric(X_patients["age"])
        X_patients["height"] = self._parse_numeric(X_patients["height"])
        X_patients["weight"] = self._parse_numeric(X_patients["weight"])
        X_patients["pulse"] = self._parse_numeric(X_patients["pulse"])

        X_patients[["BPsys", "BPdia"]] = df["BPsys/BPdia"].str.split(
            "/",
            expand=True
        )

        X_patients["BPsys"] = self._parse_numeric(X_patients["BPsys"])
        X_patients["BPdia"] = self._parse_numeric(X_patients["BPdia"])

        X_patients = pd.get_dummies(
            X_patients,
            columns=["sex"],
            drop_first=True,
            dtype=int
        )

        groups = X_patients["user_id"].astype(int).to_numpy()

        X_patients = X_patients.drop(
            columns=["user_id", "night_id"]
        )

        return X_patients, groups

    def prepare_data(self):
        df, X_wavelet = self._load_and_align_data()

        y = self._build_target(df)
        X_clinical, groups = self._build_clinical_features(df)

        valid_mask = ~np.isnan(y)

        y = y[valid_mask]
        groups = groups[valid_mask]
        X_clinical = X_clinical.iloc[valid_mask].copy()
        X_wavelet = X_wavelet[valid_mask]

        if self.mode == "clinical":
            X = X_clinical.to_numpy()

        elif self.mode == "wavelet":
            X = X_wavelet

        elif self.mode == "multimodal":
            X = np.hstack([
                X_clinical.to_numpy(),
                X_wavelet
            ])

        else:
            raise ValueError(
                "mode must be one of: 'clinical', 'wavelet', 'multimodal'"
            )

        self.X = X
        self.y = y
        self.groups = groups

        print("Mode:", self.mode)
        print("Regressor:", self.regressor_type)
        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)
        print("Groups:", len(np.unique(self.groups)), "patients")

        print("\nAHI summary:")
        print(pd.Series(self.y).describe())

        print("\nSeverity distribution after thresholding AHI:")
        print(
            pd.Series([self.ahi_to_class(v) for v in self.y])
            .value_counts()
            .sort_index()
        )

        return self.X, self.y, self.groups

    def build_model(self):
        if self.regressor_type == "ridge":
            self.model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(
                    alpha=self.alpha
                ))
            ])

        elif self.regressor_type == "random_forest":
            self.model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])

        else:
            raise ValueError(
                "regressor_type must be one of: 'ridge', 'random_forest'"
            )

        return self.model

    def cross_validate(self):
        if self.X is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        cv = GroupKFold(n_splits=self.n_splits)

        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        }

        scores = cross_validate(
            estimator=self.model,
            X=self.X,
            y=self.y,
            groups=self.groups,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        print("\nGroupKFold regression results:")

        r2_values = scores["test_r2"]
        mae_values = -scores["test_mae"]
        rmse_values = -scores["test_rmse"]

        print(f"R²   : {r2_values.mean():.3f} ± {r2_values.std():.3f}")
        print(f"MAE  : {mae_values.mean():.3f} ± {mae_values.std():.3f}")
        print(f"RMSE : {rmse_values.mean():.3f} ± {rmse_values.std():.3f}")

        return scores

    def cross_validate_thresholded_classes(self):
        """
        Evaluates the regression model as a classifier by:
            predicted AHI -> severity class.
        """

        if self.X is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        cv = GroupKFold(n_splits=self.n_splits)

        accuracies = []
        f1_macros = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(self.X, self.y, self.groups),
            start=1
        ):
            X_train = self.X[train_idx]
            X_test = self.X[test_idx]

            y_train = self.y[train_idx]
            y_test = self.y[test_idx]

            self.model.fit(X_train, y_train)

            y_pred_ahi = self.model.predict(X_test)

            y_test_class = np.array([
                self.ahi_to_class(v) for v in y_test
            ])

            y_pred_class = np.array([
                self.ahi_to_class(v) for v in y_pred_ahi
            ])

            acc = accuracy_score(
                y_test_class,
                y_pred_class
            )

            f1 = f1_score(
                y_test_class,
                y_pred_class,
                average="macro",
                zero_division=0
            )

            accuracies.append(acc)
            f1_macros.append(f1)

            print(f"\nFold {fold}")
            print(f"Accuracy after thresholding: {acc:.3f}")
            print(f"Macro F1 after thresholding: {f1:.3f}")

        print("\nRegression -> thresholded severity results:")
        print(
            f"Accuracy: {np.mean(accuracies):.3f}"
            f" ± "
            f"{np.std(accuracies):.3f}"
        )
        print(
            f"F1 macro: {np.mean(f1_macros):.3f}"
            f" ± "
            f"{np.std(f1_macros):.3f}"
        )

    def final_fold_report(self):
        if self.X is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        cv = GroupKFold(n_splits=self.n_splits)

        train_idx, test_idx = next(
            cv.split(self.X, self.y, self.groups)
        )

        X_train = self.X[train_idx]
        X_test = self.X[test_idx]

        y_train = self.y[train_idx]
        y_test = self.y[test_idx]

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        print("\nSingle-fold regression report:")
        print(f"MAE  = {mae:.3f}")
        print(f"RMSE = {rmse:.3f}")
        print(f"R²   = {r2:.3f}")

        plt.figure(figsize=(6, 6))

        plt.scatter(
            y_test,
            y_pred,
            alpha=0.7
        )

        min_val = min(
            y_test.min(),
            y_pred.min()
        )

        max_val = max(
            y_test.max(),
            y_pred.max()
        )

        plt.figure(figsize=(7, 7))

        plt.scatter(
            y_test,
            y_pred,
            color="steelblue",
            s=80,
            alpha=0.75,
            edgecolor="white",
            linewidth=1
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="darkorange",
            linewidth=3,
            linestyle="-"
        )


        plt.xlabel("True AHI")
        plt.ylabel("Predicted AHI")
        plt.title("Predicted vs True AHI")
        plt.grid(alpha=0.25)

        plt.show()

        y_test_class = np.array([
            self.ahi_to_class(v) for v in y_test
        ])

        y_pred_class = np.array([
            self.ahi_to_class(v) for v in y_pred
        ])

        print("\nClassification report after thresholding predicted AHI:")
        print(classification_report(
            y_test_class,
            y_pred_class,
            labels=[0,1,2],
            target_names=self.CLASS_NAMES,
            zero_division=0
        ))

        cm = confusion_matrix(
            y_test_class,
            y_pred_class,
            labels=[0, 1, 2]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.CLASS_NAMES
        )

        disp.plot()
        plt.title("Confusion Matrix: regression -> severity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self):
        self.prepare_data()
        self.build_model()
        self.cross_validate()
        self.cross_validate_thresholded_classes()
        self.final_fold_report()


def main():
    experiment = MultimodalAHIRegressor(
        patients_csv_path=(
            r"C:\Users\tomma\.cache\kagglehub\datasets"
            r"\yfrite\polysom\versions\3\patients.csv"
        ),
        wavelet_embeddings_path="wavelet_embeddings.npy",
        completed_samples_path="wavelet_completed_samples.npy",
        mode="multimodal",           # "clinical", "wavelet", "multimodal"
        regressor_type="ridge",      # "ridge", "random_forest"
        n_splits=5,
        alpha=10.0,
        n_estimators=500,
        max_depth=5,
        random_state=42
    )

    experiment.run()


if __name__ == "__main__":
    main()