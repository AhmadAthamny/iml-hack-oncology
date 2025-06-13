import pandas as pd
from os import path
from sklearn.ensemble import RandomForestClassifier

from utils import split_train_test, run_estimator
from preprocessing import (
    load_data_question_1,
    DATA_PATH,
    TRAIN_FILE,
    LABELS_FILE_1,
    TEST_FILE,
    LABELS_COL
)

PRED_CSV = "pred.csv"
LABEL_OPTIONS = [
    'BON - Bones', 'LYM - Lymph nodes', 'HEP - Hepatic',
    'PUL - Pulmonary', 'PLE - Pleura', 'SKI - Skin',
    'OTH - Other', 'BRA - Brain', 'MAR - Bone Marrow',
    'PER - Peritoneum', 'ADR - Adrenals'
]


def encode_labels(y_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode multilabel targets for classification.
    """
    encoded = pd.DataFrame({
        label: y_df[LABELS_COL].apply(lambda s: 1 if label in s else 0)
        for label in LABEL_OPTIONS
    })
    return encoded


def main():
    # Load data
    features, labels = load_data_question_1(
        path.join(DATA_PATH, TRAIN_FILE),
        path.join(DATA_PATH, LABELS_FILE_1)
    )
    X_train, y_train, _, _ = split_train_test(features, labels)
    
    # Prepare test features
    X_test, _ = load_data_question_1(
        path.join(DATA_PATH, TEST_FILE)
    )

    # Train and predict for each label
    predictions = pd.DataFrame(
        {
            label: run_estimator(
                RandomForestClassifier,
                X_train,
                encode_labels(labels)[label],
                X_test,
                n_estimators=100,
                random_state=1
            )
            for label in LABEL_OPTIONS
        }
    )

    # Convert to multilabel strings
    out = pd.DataFrame(
        predictions.apply(lambda row: str([lbl for lbl, val in row.items() if val > 0]), axis=1),
        columns=[LABELS_COL]
    )
    out.to_csv(PRED_CSV, index=False)


if __name__ == "__main__":
    main()
