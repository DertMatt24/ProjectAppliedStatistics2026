import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from loader.patients import PatientsCSVLoader
from sklearn.model_selection import GroupShuffleSplit
import plotting
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)


WAVELET_EMBEDDINGS_PATH = "wavelet_embeddings.npy"
COMPLETED_SAMPLES_PATH = "wavelet_completed_samples.npy"

PATIENTS_CSV_PATH = (
    r"C:\Users\tomma\.cache\kagglehub\datasets"
    r"\yfrite\polysom\versions\3\patients.csv"
)


AHI_THRESHOLD = 5.0




def main():
    X_wavelet = np.load(WAVELET_EMBEDDINGS_PATH)
    completed_samples = np.load(COMPLETED_SAMPLES_PATH)
    print(X_wavelet.shape)
    print(X_wavelet.shape[1])
    # This dataframe has exactly the same row order as X_wavelet
    samples_df = pd.DataFrame(
        completed_samples,
        columns=["user_id", "night_id"]
    )

    samples_df["user_id"] = pd.to_numeric(samples_df["user_id"], errors="coerce")
    samples_df["night_id"] = pd.to_numeric(samples_df["night_id"], errors="coerce")

    df = PatientsCSVLoader.load_dataframe(PATIENTS_CSV_PATH)

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["night_id"] = pd.to_numeric(df["night_id"], errors="coerce")

    # IMPORTANT:
    # merge clinical data using completed_samples as master order
    df = samples_df.merge(
        df,
        on=["user_id", "night_id"],
        how="left"
    )

    # Safety checks
    if len(df) != len(X_wavelet):
        raise ValueError(
            f"Alignment error: df has {len(df)} rows, "
            f"X_wavelet has {len(X_wavelet)} rows"
        )

    df["AHI"] = pd.to_numeric(df["AHI"], errors="coerce").fillna(0)

    df["severity"] = pd.cut(
        x=df["AHI"],
        bins=[-np.inf, 5, 15, 30, np.inf],
        labels=["healthy", "mild", "moderate", "severe"],
        right=False
    )

    y = df["severity"]

    X_patients = df[[
        "user_id",
        "night_id",
        "age",
        "sex",
        "height",
        "weight",
        "pulse"
    ]].copy()

    X_patients["age"] = pd.to_numeric(X_patients["age"], errors="coerce")
    X_patients["height"] = pd.to_numeric(X_patients["height"], errors="coerce")
    X_patients["weight"] = pd.to_numeric(X_patients["weight"], errors="coerce")
    X_patients["pulse"] = pd.to_numeric(X_patients["pulse"], errors="coerce")

    X_patients[["BPsys", "BPdia"]] = df["BPsys/BPdia"].str.split(
        "/",
        expand=True
    )

    X_patients["BPsys"] = pd.to_numeric(X_patients["BPsys"], errors="coerce")
    X_patients["BPdia"] = pd.to_numeric(X_patients["BPdia"], errors="coerce")

    X_patients = pd.get_dummies(
        X_patients,
        columns=["sex"],
        drop_first=True,
        dtype=int
    )

    groups = X_patients["user_id"].astype(int).to_numpy()

    # Drop identifiers after creating groups
    X_patients = X_patients.drop(
        columns=["user_id", "night_id"]
    )

    mode = "wavelet"

    if mode == "wavelet":
        X = np.hstack([
            X_patients.to_numpy(),
            X_wavelet
        ])
    else:
        X = X_patients.to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=6884
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y.iloc[train_idx].cat.codes
    y_test = y.iloc[test_idx].cat.codes

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = sm.add_constant(X_train_scaled)
    X_test_scaled = sm.add_constant(X_test_scaled)

    model = sm.MNLogit(y_train, X_train_scaled)

    result = model.fit_regularized(
        alpha=1.0,
        method="l1",
        disp=True
    )

    print(result.summary())

    y_test_pred_prob = result.predict(X_test_scaled)

    y_test_pred_class = np.asarray(y_test_pred_prob).argmax(axis=1)

    cm = confusion_matrix(
        y_test,
        y_test_pred_class,
        labels=[0, 1, 2, 3]
    )

    class_names = ["healthy", "mild", "moderate", "severe"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    class_names = ["healthy", "mild", "moderate", "severe"]

    y_test_bin = label_binarize(
        y_test,
        classes=[0, 1, 2, 3]
    )

    auc_scores = {}

    for i, class_name in enumerate(class_names):
        # AUC is undefined if the test set contains only positives or only negatives for this class
        if len(np.unique(y_test_bin[:, i])) < 2:
            print(f"{class_name:10s}: AUC undefined, class absent or alone in test set")
            continue

        auc = roc_auc_score(
            y_test_bin[:, i],
            y_test_pred_prob[:, i]
        )

        auc_scores[class_name] = auc
        print(f"{class_name:10s}: AUC = {auc:.3f}")

if __name__ == "__main__":
    main()
