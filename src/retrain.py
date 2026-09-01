import subprocess
import sys

import pandas as pd

from drift import (
    calculate_numeric_drift,
    calculate_categorical_drift,
)


def check_drift():
    # Reference dataset
    reference = pd.read_csv(
        "data/bank-full.csv",
        sep=";",
    )

    # Current / incoming dataset
    current = pd.read_csv(
        "data/current_data.csv",
        sep=";",
    )

    # Numeric drift
    numeric_results = calculate_numeric_drift(
        reference,
        current,
    )

    # Categorical drift
    categorical_results = calculate_categorical_drift(
        reference,
        current,
    )

    results = (
        numeric_results
        + categorical_results
    )

    # Ignore target-column drift because
    # the target is an outcome, not an input feature.
    drift_count = sum(
        result["drift_detected"]
        for result in results
        if result["feature"] != "y"
    )

    return drift_count


def main():

    print("Checking data drift...")

    drift_count = check_drift()

    print(
        f"Drifted features: {drift_count}"
    )

    if drift_count > 0:

        print(
            "Drift detected!"
        )

        print(
            "Starting model retraining..."
        )

        # Use the same Python interpreter
        # that is running this script.
        subprocess.run(
            [
                sys.executable,
                "src/train.py",
            ],
            check=True,
        )

        print(
            "\nRetraining completed successfully."
        )

    else:

        print(
            "No significant drift detected."
        )

        print(
            "Retraining is not required."
        )


if __name__ == "__main__":
    main()