import os
import numpy as np
import pandas as pd

# ——— File and directory constants ———
DATA_DIR           = "Mission 2 - Breast Cancer"
TRAIN_FEATURES     = "train.feats.csv"
TEST_FEATURES      = "test.feats.csv"
LABELS_FILE_Q1     = "train.labels.0.csv"
LABELS_FILE_Q2     = "train.labels.1.csv"

# ——— Column name constants ———
DATE_COLUMNS       = [
    "אבחנה-Diagnosis date",
    "אבחנה-Surgery date1",
    "אבחנה-Surgery date2",
    "אבחנה-Surgery date3",
]
SURGERY_SUMMARY    = "אבחנה-Surgery sum"
SURGERY_DATES      = ["אבחנה-Surgery date1", "אבחנה-Surgery date2", "אבחנה-Surgery date3"]
NODE_EXAM_COL      = "אבחנה-Nodes exam"
POSITIVE_NODE_COL  = "אבחנה-Positive nodes"
PATIENT_ID_COL     = "id-hushed_internalpatientid"
LABELS_COL         = "labels"
TODAY_MARKER       = "today"
STAGE_COL          = "אבחנה-Stage"

# Columns we don’t need for either question
COLUMNS_TO_DROP = [
    " Hospital", "אבחנה-Her2", "אבחנה-Histopatological degree",
    "אבחנה-Ivi -Lymphovascular invasion", "אבחנה-KI67 protein",
    "אבחנה-Lymphatic penetration", "אבחנה-M -metastases mark (TNM)",
    "אבחנה-Margin Type", "אבחנה-N -lymph nodes mark (TNM)",
    "אבחנה-Surgery name1", "אבחנה-Surgery name2", "אבחנה-Surgery name3",
    "אבחנה-T -Tumor mark (TNM)", "אבחנה-Tumor depth", "אבחנה-Tumor width",
    "surgery before or after-Actual activity", "surgery before or after-Activity date",
    "אבחנה-er", "אבחנה-pr"
]

# Mapping for staging
STAGE_MAPPING = {
    "Stage0": 0, "Stage0is": 0, "Stage0a": 0,
    "Stage1": 1, "LA": 1, "Stage1a": 1,
    "Stage1b": 2, "Stage1c": 3,
    "Stage2": 4, "Stage2a": 4, "Stage2b": 5,
    "Stage3": 6, "Stage3a": 6, "Stage 3a": 6,
    "Stage3b": 7, "Stage 3b": 7,
    "Stage3c": 8, "Stage 3c": 8,
    "Stage4": 9, "Stage 4": 9,
    "Not yet Established": 0
}


def process_surgeries(df: pd.DataFrame) -> None:
    """Convert surgery dates to datetime and summary to numeric."""
    df[SURGERY_DATES] = df[SURGERY_DATES].apply(pd.to_datetime, errors="coerce")
    df[SURGERY_SUMMARY] = pd.to_numeric(df[SURGERY_SUMMARY], errors="coerce")


def normalize_receptors(cell: str) -> int:
    """Binary-encode ER/PR receptor status."""
    val = str(cell).strip().lower()
    return 0 if val in {"no", "-", ""} else 1


def extract_name_age(df: pd.DataFrame) -> None:
    """Pull out User Name & Age into dedicated columns."""
    df["user_name"]      = df.pop("User Name")
    df["age_at_diagnosis"] = pd.to_numeric(df.pop("אבחנה-Age"), errors="coerce")


def encode_basic_stage(df: pd.DataFrame) -> None:
    """Map basic-stage strings into numeric codes."""
    df["basic_stage_code"] = (
        df.pop("אבחנה-Basic stage")
          .map(STAGE_MAPPING)
          .fillna(0)
          .astype(int)
    )


def clean_diagnosis_text(df: pd.DataFrame) -> None:
    """Standardize histological diagnosis strings."""
    df["histological_diagnosis"] = (
        df.pop("אבחנה-Histological diagnosis")
          .str.strip()
          .str.lower()
    )


def adjust_node_counts(df: pd.DataFrame) -> None:
    """Ensure node counts are integer and NaNs → 0."""
    df[NODE_EXAM_COL]     = pd.to_numeric(df[NODE_EXAM_COL], errors="coerce").fillna(0).astype(int)
    df[POSITIVE_NODE_COL] = pd.to_numeric(df[POSITIVE_NODE_COL], errors="coerce").fillna(0).astype(int)


def rename_patient_id(df: pd.DataFrame) -> None:
    """Rename the ID column for clarity."""
    if PATIENT_ID_COL in df.columns:
        df.rename(columns={PATIENT_ID_COL: "patient_id"}, inplace=True)


def fill_missing(df: pd.DataFrame) -> None:
    """Fill NaNs and Infs uniformly."""
    df.fillna("Unknown", inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)


def load_data_question_1(feature_path: str, labels_path: str = None):
    df = pd.read_csv(feature_path, dtype="unicode")
    if labels_path:
        df[LABELS_COL] = pd.read_csv(labels_path, dtype="unicode")

    # clean-up
    df.drop(columns=COLUMNS_TO_DROP, errors="ignore", inplace=True)
    df[TODAY_MARKER] = pd.to_datetime(TODAY_MARKER, errors="coerce")

    process_surgeries(df)
    extract_name_age(df)
    encode_basic_stage(df)
    clean_diagnosis_text(df)
    adjust_node_counts(df)
    rename_patient_id(df)
    fill_missing(df)

    labels = df.pop(LABELS_COL)
    return df, labels


def load_data_question_2(feature_path: str, labels_path: str = None):
    df = pd.read_csv(feature_path, dtype="unicode")
    if labels_path:
        df[LABELS_COL] = pd.read_csv(labels_path, dtype="unicode")

    # clean-up
    df.drop(columns=COLUMNS_TO_DROP, errors="ignore", inplace=True)
    df[TODAY_MARKER] = pd.to_datetime(TODAY_MARKER, errors="coerce")

    process_surgeries(df)
    extract_name_age(df)
    encode_basic_stage(df)
    clean_diagnosis_text(df)
    rename_patient_id(df)
    fill_missing(df)

    # question 2 doesn’t need node info
    df.drop(columns=[NODE_EXAM_COL, POSITIVE_NODE_COL], errors="ignore", inplace=True)

    labels = df.pop(LABELS_COL)
    return df, labels


def main():
    base = DATA_DIR
    f_train = os.path.join(base, TRAIN_FEATURES)
    f_test  = os.path.join(base, TEST_FEATURES)

    # Q1
    df1, lbl1 = load_data_question_1(
        os.path.join(base, TRAIN_FEATURES),
        os.path.join(base, LABELS_FILE_Q1)
    )

    # Q2
    df2, lbl2 = load_data_question_2(
        os.path.join(base, TRAIN_FEATURES),
        os.path.join(base, LABELS_FILE_Q2)
    )


if __name__ == "__main__":
    main()
