import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader.patients import PatientsCSVLoader

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


class ClassificationExperiment:

    def __init__(self, embeddings_path, patients_csv_path):
        self.embeddings_path = embeddings_path
        self.patients_csv_path = patients_csv_path

        self.X = None
        self.y = None
        self.attack_counts = None
        self.samples = None

        self.models = self._build_models()

    def build_samples(self):
        """
        Build all patient-night pairs.

        Dataset:
            40 patients
            2 nights each

        Remove problematic recordings.
        """

        samples = []

        for patient_id in range(1, 40):
            for night_id in range(1, 3):
                samples.append((patient_id, night_id))

        samples.remove((8, 1))
        samples.remove((14, 2))

        return samples

    def load_attack_counts(self, samples):
        """
        Load NAp values from patients.csv.

        NAp = number of apnea events.

        Missing values are interpreted as 0.
        """

        df = PatientsCSVLoader.load_dataframe(self.patients_csv_path)

        attack_counts = []

        for patient_id, night_id in samples:

            row = df[
                (df["user_id"] == patient_id)
                &
                (df["night_id"] == night_id)
            ]

            attacks = row.iloc[0]["NAp"]

            if pd.isna(attacks) or attacks == "" or attacks == "NaN":
                attacks = 0
            else:
                attacks = int(attacks)

            attack_counts.append(attacks)

        return np.array(attack_counts)

    def apnea_to_class(self, attacks):
        if attacks == 0:
            return 0  # no apnea
        elif attacks < 30:
            return 1  # mild/moderate
        else:
            return 2  # severe

    def load_data(self):
        """
        Load embeddings and build multiclass labels.
        """

        self.samples = self.build_samples()

        self.X = np.load(self.embeddings_path)

        self.attack_counts = self.load_attack_counts(self.samples)

        self.y = np.array([
            self.apnea_to_class(attacks)
            for attacks in self.attack_counts
        ])

        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)

        if len(self.X) != len(self.y):
            raise ValueError(
                f"Mismatch: X has {len(self.X)} rows, "
                f"but y has {len(self.y)} labels"
            )

        print("\nClass distribution:")
        print(pd.Series(self.y).value_counts().sort_index())

    def _build_models(self):
        """
        Build all classification models.
        """

        return {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced"
                ))
            ]),

            "k-NN": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(
                    n_neighbors=5
                ))
            ]),

            "Decision Tree": DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced"
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced"
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42
            ),

            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(
                    kernel="rbf",
                    C=1.0,
                    class_weight="balanced"
                ))
            ])
        }

    def cross_validate_models(self):
        """
        Evaluate all models using stratified 5-fold cross-validation.
        """

        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        )

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro"
        }

        results = []

        for name, model in self.models.items():

            cv_results = cross_validate(
                model,
                self.X,
                self.y,
                cv=cv,
                scoring=scoring,
                return_train_score=True
            )

            results.append({
                "model": name,

                "accuracy_mean": cv_results["test_accuracy"].mean(),
                "accuracy_std": cv_results["test_accuracy"].std(),

                "precision_mean": cv_results["test_precision"].mean(),
                "precision_std": cv_results["test_precision"].std(),

                "recall_mean": cv_results["test_recall"].mean(),
                "recall_std": cv_results["test_recall"].std(),

                "f1_mean": cv_results["test_f1"].mean(),
                "f1_std": cv_results["test_f1"].std(),

                "train_f1_mean": cv_results["train_f1"].mean()
            })

            print(f"\n{name}")
            print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
            print(f"F1 macro: {cv_results['test_f1'].mean():.4f}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("f1_mean", ascending=False)

        print("\nCross-validation comparison:")
        print(results_df)

        return results_df

    def train_test_report(self, model_name):
        """
        Train the selected model on a train/test split
        and show classification report + confusion matrix.
        """

        model = self.models[model_name]

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(f"\nFinal report for: {model_name}")
        print(classification_report(
            y_test,
            y_pred,
            target_names=[
                "No apnea",
                "Mild/Moderate",
                "Severe"
            ],
            zero_division=0
        ))

        cm = confusion_matrix(y_test, y_pred)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "No apnea",
                "Mild/Moderate",
                "Severe"
            ]
        )

        display.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()


def main():

    experiment = ClassificationExperiment(
        embeddings_path=(
            r"C:\Users\tomma\.cache\kagglehub\datasets"
            r"\yfrite\polysom\versions\3\Embeddings"
            r"\embeddings.npy"
        ),
        patients_csv_path=(
            r"C:\Users\tomma\.cache\kagglehub\datasets"
            r"\yfrite\polysom\versions\3\patients.csv"
        )
    )

    experiment.load_data()

    results = experiment.cross_validate_models()

    best_model_name = results.iloc[0]["model"]

    print("\nBest model:", best_model_name)

    experiment.train_test_report(best_model_name)


if __name__ == "__main__":
    main()