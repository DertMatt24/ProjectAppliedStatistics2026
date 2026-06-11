import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader.patients import PatientsCSVLoader

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


class KNNFeatureSelectionWaveletClassifier:
    """
    Feature selection + KNN classifier using wavelet embeddings.

    Classes:
        0 = healthy
        1 = moderate
        2 = severe
    """

    CLASS_NAMES = ["healthy", "moderate", "severe"]

    def __init__(
        self,
        patients_csv_path,
        wavelet_embeddings_path,
        completed_samples_path,
        mode="multimodal",
        n_splits=5,
        k_features=20,
        n_neighbors=5,
        weights="distance"
    ):
        self.patients_csv_path = patients_csv_path
        self.wavelet_embeddings_path = wavelet_embeddings_path
        self.completed_samples_path = completed_samples_path

        self.mode = mode
        self.n_splits = n_splits
        self.k_features = k_features
        self.n_neighbors = n_neighbors
        self.weights = weights

        self.X = None
        self.y = None
        self.groups = None
        self.model = None

    @staticmethod
    def _parse_numeric(series):
        return pd.to_numeric(series, errors="coerce")

    def _load_wavelet_embedding(self):
        X_wavelet = np.load(self.wavelet_embeddings_path)

        print("Original wavelet embedding shape:", X_wavelet.shape)

        if X_wavelet.ndim == 3:
            X_wavelet = X_wavelet.reshape(X_wavelet.shape[0], -1)

        print("Final wavelet embedding shape:", X_wavelet.shape)

        return X_wavelet

    def _load_and_align_data(self):
        X_wavelet = self._load_wavelet_embedding()
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
                f"wavelet embedding has {len(X_wavelet)} rows."
            )

        alignment_ok = (
            df[["user_id", "night_id"]].to_numpy()
            == completed_samples
        ).all()

        if not alignment_ok:
            raise ValueError("Row alignment between df and wavelet embeddings failed.")

        return df, X_wavelet

    def _build_target(self, df):
        df["AHI"] = pd.to_numeric(df["AHI"], errors="coerce").fillna(0)

        df["severity"] = pd.cut(
            x=df["AHI"],
            bins=[-np.inf, 5, 30, np.inf],
            labels=self.CLASS_NAMES,
            right=False
        )

        return df["severity"].cat.codes.to_numpy()

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
        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)
        print("Groups:", len(np.unique(self.groups)), "patients")

        print("\nClass distribution:")
        print(pd.Series(self.y).value_counts().sort_index())

        return self.X, self.y, self.groups

    def build_model(self):
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),

            ("selector", SelectKBest(
                score_func=f_classif,
                k=self.k_features
            )),

            ("classifier", KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                metric="minkowski",
                p=2
            ))
        ])

        return self.model

    def cross_validate(self):
        if self.X is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        cv = GroupKFold(n_splits=self.n_splits)

        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
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

        print("\nGroupKFold cross-validation results:")
        for metric in scoring:
            values = scores[f"test_{metric}"]
            print(f"{metric:20s}: {values.mean():.3f} ± {values.std():.3f}")

        return scores

    def aggregated_report(self):
        if self.X is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        cv = GroupKFold(n_splits=self.n_splits)

        all_y_true = []
        all_y_pred = []
        selected_counts = []

        for train_idx, test_idx in cv.split(self.X, self.y, self.groups):
            X_train = self.X[train_idx]
            X_test = self.X[test_idx]

            y_train = self.y[train_idx]
            y_test = self.y[test_idx]

            self.model.fit(X_train, y_train)

            selector = self.model.named_steps["selector"]
            selected_counts.append(selector.get_support().sum())

            y_pred = self.model.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        print("\nAverage selected features:")
        print(f"{np.mean(selected_counts):.1f} ± {np.std(selected_counts):.1f}")

        print("\nAggregated classification report:")
        print(classification_report(
            all_y_true,
            all_y_pred,
            labels=[0, 1, 2],
            target_names=self.CLASS_NAMES,
            zero_division=0
        ))

        cm = confusion_matrix(
            all_y_true,
            all_y_pred,
            labels=[0, 1, 2]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.CLASS_NAMES
        )

        disp.plot()
        plt.title(f"KNN + Feature Selection - Wavelet - {self.mode}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self):
        self.prepare_data()
        self.build_model()
        self.cross_validate()
        self.aggregated_report()


def main():
    experiment = KNNFeatureSelectionWaveletClassifier(
        patients_csv_path=(
            r"C:\Users\tomma\.cache\kagglehub\datasets"
            r"\yfrite\polysom\versions\3\patients.csv"
        ),
        wavelet_embeddings_path="wavelet_embeddings.npy",
        completed_samples_path="wavelet_completed_samples.npy",
        mode="multimodal",      # "clinical", "wavelet", "multimodal"
        n_splits=5,
        k_features=20,
        n_neighbors=5,
        weights="distance"
    )

    experiment.run()


if __name__ == "__main__":
    main()