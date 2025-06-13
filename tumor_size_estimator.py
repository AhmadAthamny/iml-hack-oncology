import pandas as pd
from os import path
from sklearn.linear_model import Ridge

from utils import split_train_test, run_estimator
from preprocessing import (
    load_data_question_2,
    DATA_PATH,
    TRAIN_FILE,
    LABELS_FILE_2,
    TEST_FILE
)

LABEL_COLUMN = "labels"
OUTPUT_CSV = "prediction.csv"


def main():
    # Load and split training data
    features, target = load_data_question_2(
        path.join(DATA_PATH, TRAIN_FILE),
        path.join(DATA_PATH, LABELS_FILE_2)
    )
    X_train, y_train, _, _ = split_train_test(features, target)

    # Load test features
    X_test, _ = load_data_question_2(
        path.join(DATA_PATH, TEST_FILE)
    )

    # Train and predict
    preds = run_estimator(
        Ridge,
        X_train,
        y_train,
        X_test,
        alpha=0.05
    )

    # Save output
    pd.DataFrame({LABEL_COLUMN: preds}).to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()