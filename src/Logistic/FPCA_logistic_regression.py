import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader.patients import PatientsCSVLoader

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


class WindowedPowerSeverityClassifier:
    CLASS_NAMES = ["healthy", "moderate", "severe"]

    def __init__(
        self,
        patients_csv_path,
        embedding_path,
        completed_samples_path,
        mode="embedding",
        n_splits=5,
        C=1.0,
        random_state=42
    ):
        self.patients_csv_path = patients_csv_path
        self.embedding_path = embedding_path
        self.completed_samples_path = completed_samples_path
        self.mode = mode
        self.n_splits = n_splits
        self.C = C
        self.random_state = random_state

        self.X = None
        self.y = None
        self.groups = None
        self.model = None

    @staticmethod
    def _parse_numeric(series):
        return pd.to_numeric(series, errors="coerce")

    def _load_embedding(self):
        X_emb = np.load(self.embedding_path)

        print("Original embedding shape:", X_emb.shape)

        if X_emb.ndim == 3:
            X_emb = X_emb.reshape(X_emb.shape[0], -1)

        print("Flattened embedding shape:", X_emb.shape)

        return X_emb

    def _load_and_align_data(self):
        X_emb = self._load_embedding()
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
            raise ValueError("Row alignment failed.")

        return df, X_emb

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

        X_patients = X_patients.drop(columns=["user_id", "night_id"])

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

        print("\nClass distribution:")
        print(pd.Series(self.y).value_counts().sort_index())

        return self.X, self.y, self.groups

    def build_model(self):
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                penalty="l2",
                C=self.C,
                class_weight="balanced"
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

        print("\nClassification report on one GroupKFold split:")
        print(classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=self.CLASS_NAMES,
            zero_division=0
        ))

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.CLASS_NAMES
        )

        disp.plot()
        plt.title(f"Confusion Matrix - {self.mode}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self):
        self.prepare_data()
        self.build_model()
        self.cross_validate()
        self.final_fold_report()


def main():
    experiment = WindowedPowerSeverityClassifier(
        patients_csv_path=(
            r"C:\Users\tomma\.cache\kagglehub\datasets"
            r"\yfrite\polysom\versions\3\patients.csv"
        ),
        embedding_path="windowed_power_embeddings.npy",
        completed_samples_path="wavelet_completed_samples.npy",
        mode="multimodal",  # "clinical", "embedding", "multimodal"
        n_splits=5,
        C=1.0
    )

    experiment.run()


if __name__ == "__main__":
    main()