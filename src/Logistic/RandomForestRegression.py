import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader.patients import PatientsCSVLoader

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


class RandomForestAHIRegressorWithPCA:
    CLASS_NAMES = ["healthy", "moderate", "severe"]

    def __init__(
        self,
        patients_csv_path,
        embedding_path,
        completed_samples_path,
        mode="multimodal",
        n_splits=5,
        n_estimators=500,
        max_depth=5,
        n_components=20,
        random_state=42
    ):
        self.patients_csv_path = patients_csv_path
        self.embedding_path = embedding_path
        self.completed_samples_path = completed_samples_path

        self.mode = mode
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_components = n_components
        self.random_state = random_state

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

    def _load_embedding(self):
        X_emb = np.load(self.embedding_path)

        print("Original embedding shape:", X_emb.shape)

        if X_emb.ndim == 3:
            X_emb = X_emb.reshape(X_emb.shape[0], -1)

        print("Final embedding shape:", X_emb.shape)

        return X_emb

    def _load_and_align_data(self):
        X_emb = self._load_embedding()
        completed_samples = np.load(self.completed_samples_path)

        samples_df = pd.DataFrame(
            completed_samples,
            columns=["user_id", "night_id"]
        )

        samples_df["user_id"] = pd.to_numeric(
            samples_df["user_id"],
            errors="coerce"
        )

        samples_df["night_id"] = pd.to_numeric(
            samples_df["night_id"],
            errors="coerce"
        )

        df = PatientsCSVLoader.load_dataframe(self.patients_csv_path)

        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
        df["night_id"] = pd.to_numeric(df["night_id"], errors="coerce")

        df = samples_df.merge(
            df,
            on=["user_id", "night_id"],
            how="left"
        )

        if len(df) != len(X_emb):
            raise ValueError(
                f"Alignment error: df has {len(df)} rows, "
                f"embedding has {len(X_emb)} rows."
            )

        alignment_ok = (
            df[["user_id", "night_id"]].to_numpy()
            == completed_samples
        ).all()

        if not alignment_ok:
            raise ValueError(
                "Row alignment between dataframe and embedding failed."
            )

        return df, X_emb

    def _build_target(self, df):
        ahi = (
            df["AHI"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )

        ahi = pd.to_numeric(
            ahi,
            errors="coerce"
        ).fillna(0)

        print("Rows:", len(ahi))
        print("AHI equal to 0:", (ahi == 0).sum())

        return ahi.to_numpy(dtype=float)

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
        df, X_emb = self._load_and_align_data()

        y = self._build_target(df)
        X_clinical, groups = self._build_clinical_features(df)

        if self.mode == "clinical":
            X = X_clinical.to_numpy()

        elif self.mode == "embedding":
            X = X_emb

        elif self.mode == "multimodal":
            X = np.hstack([
                X_clinical.to_numpy(),
                X_emb
            ])

        else:
            raise ValueError(
                "mode must be one of: 'clinical', 'embedding', 'multimodal'"
            )

        self.X = X
        self.y = y
        self.groups = groups

        print("Mode:", self.mode)
        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)
        print("Groups:", len(np.unique(self.groups)), "patients")

        print("\nAHI summary:")
        print(pd.Series(self.y).describe())

        print("\nSeverity distribution after thresholding AHI:")
        print(
            pd.Series([
                self.ahi_to_class(v)
                for v in self.y
            ])
            .value_counts()
            .sort_index()
        )

        return self.X, self.y, self.groups

    def build_model(self):
        regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),

            ("scaler", StandardScaler()),

            ("pca", PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )),

            ("regressor", regressor)
        ])

        return self.model

    def cross_validate(self):
        if self.X is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        cv = GroupKFold(n_splits=self.n_splits)

        r2_scores = []
        mae_scores = []
        rmse_scores = []

        acc_scores = []
        f1_scores = []

        explained_variance_scores = []

        all_true_classes = []
        all_pred_classes = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(self.X, self.y, self.groups),
            start=1
        ):
            X_train = self.X[train_idx]
            X_test = self.X[test_idx]

            y_train = self.y[train_idx]
            y_test = self.y[test_idx]

            self.model.fit(X_train, y_train)

            pca = self.model.named_steps["pca"]
            explained_variance = pca.explained_variance_ratio_.sum()
            explained_variance_scores.append(explained_variance)

            y_pred_ahi = self.model.predict(X_test)

            r2 = r2_score(y_test, y_pred_ahi)
            mae = mean_absolute_error(y_test, y_pred_ahi)
            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred_ahi
                )
            )

            y_test_class = np.array([
                self.ahi_to_class(v)
                for v in y_test
            ])

            y_pred_class = np.array([
                self.ahi_to_class(v)
                for v in y_pred_ahi
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

            r2_scores.append(r2)
            mae_scores.append(mae)
            rmse_scores.append(rmse)

            acc_scores.append(acc)
            f1_scores.append(f1)

            all_true_classes.extend(y_test_class)
            all_pred_classes.extend(y_pred_class)

            print(f"\nFold {fold}")
            print(f"PCA components: {self.n_components}")
            print(f"Explained variance: {explained_variance:.3f}")
            print(f"R²: {r2:.3f}")
            print(f"MAE: {mae:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"Thresholded accuracy: {acc:.3f}")
            print(f"Thresholded macro F1: {f1:.3f}")

        print("\nGroupKFold Random Forest regression results:")
        print(f"R²   : {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"MAE  : {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
        print(f"RMSE : {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")

        print("\nRegression -> thresholded severity:")
        print(f"Accuracy : {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
        print(f"Macro F1 : {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

        print("\nPCA:")
        print(
            f"Explained variance: "
            f"{np.mean(explained_variance_scores):.3f} "
            f"± {np.std(explained_variance_scores):.3f}"
        )

        print("\nAggregated classification report:")
        print(classification_report(
            all_true_classes,
            all_pred_classes,
            labels=[0, 1, 2],
            target_names=self.CLASS_NAMES,
            zero_division=0
        ))

        cm = confusion_matrix(
            all_true_classes,
            all_pred_classes,
            labels=[0, 1, 2]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.CLASS_NAMES
        )

        disp.plot()
        plt.title("Aggregated confusion matrix: RF regression + PCA -> severity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def final_fold_regression_plot(self):
        if self.X is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        cv = GroupKFold(n_splits=self.n_splits)

        train_idx, test_idx = next(
            cv.split(
                self.X,
                self.y,
                self.groups
            )
        )

        X_train = self.X[train_idx]
        X_test = self.X[test_idx]

        y_train = self.y[train_idx]
        y_test = self.y[test_idx]

        self.model.fit(
            X_train,
            y_train
        )

        y_pred = self.model.predict(
            X_test
        )

        plt.figure(figsize=(8, 8))

        plt.scatter(
            y_test,
            y_pred,
            color="royalblue",
            edgecolors="black",
            linewidth=0.5,
            alpha=0.75,
            s=70,
            label="Patients"
        )

        min_val = 0

        max_val = max(
            y_test.max(),
            y_pred.max()
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="crimson",
            linestyle="--",
            linewidth=3,
            label="Perfect prediction"
        )

        plt.xlabel("True AHI")
        plt.ylabel("Predicted AHI")
        plt.title("Random Forest Regression + PCA\nPredicted vs True AHI")

        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def pca_report_on_full_data(self):
        """
        Fits the model on the full dataset only to inspect PCA variance.
        Do not use this for performance evaluation.
        """

        if self.X is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        self.model.fit(
            self.X,
            self.y
        )

        pca = self.model.named_steps["pca"]

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        print("\nPCA full-data report:")
        print(f"Components: {self.n_components}")
        print(f"Explained variance: {cumulative[-1]:.3f}")

        plt.figure(figsize=(8, 5))

        plt.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            marker="o",
            linewidth=2,
            color="royalblue",
            label="Cumulative explained variance"
        )

        plt.axhline(
            0.90,
            linestyle="--",
            linewidth=2,
            color="crimson",
            label="90% variance"
        )

        plt.xlabel("Number of PCA components")
        plt.ylabel("Cumulative explained variance")
        plt.title("PCA explained variance")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        self.prepare_data()
        self.build_model()
        self.cross_validate()
        self.final_fold_regression_plot()
        self.pca_report_on_full_data()


def main():
    experiment = RandomForestAHIRegressorWithPCA(
        patients_csv_path=(
            r"C:\Users\tomma\.cache\kagglehub\datasets"
            r"\yfrite\polysom\versions\3\patients.csv"
        ),
        embedding_path="wavelet_embeddings.npy",
        completed_samples_path="wavelet_completed_samples.npy",
        mode="multimodal",
        n_splits=5,
        n_estimators=500,
        max_depth=5,
        n_components=20,
        random_state=42
    )

    experiment.run()


if __name__ == "__main__":
    main()