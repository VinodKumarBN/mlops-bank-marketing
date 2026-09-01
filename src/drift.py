import pandas as pd
import numpy as np


def calculate_numeric_drift(reference, current):
    results = []

    numeric_features = reference.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numeric_features:
        ref_mean = reference[column].mean()
        cur_mean = current[column].mean()

        if ref_mean == 0:
            drift_score = abs(cur_mean - ref_mean)
        else:
            drift_score = abs(cur_mean - ref_mean) / abs(ref_mean)

        results.append({
            "feature": column,
            "type": "numeric",
            "drift_score": drift_score,
            "drift_detected": drift_score > 0.20
        })

    return results


def calculate_categorical_drift(reference, current):
    results = []

    categorical_features = reference.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_features:
        ref_distribution = reference[column].value_counts(
            normalize=True
        )

        cur_distribution = current[column].value_counts(
            normalize=True
        )

        all_categories = set(ref_distribution.index) | set(cur_distribution.index)

        drift_score = 0.0

        for category in all_categories:
            ref_value = ref_distribution.get(category, 0)
            cur_value = cur_distribution.get(category, 0)

            drift_score += abs(ref_value - cur_value)

        results.append({
            "feature": column,
            "type": "categorical",
            "drift_score": drift_score,
            "drift_detected": drift_score > 0.20
        })

    return results


def main():
    reference = pd.read_csv("data/bank-full.csv", sep=";")

    current = pd.read_csv("data/current_data.csv", sep=";")

    numeric_results = calculate_numeric_drift(reference, current)
    categorical_results = calculate_categorical_drift(reference, current)

    results = numeric_results + categorical_results

    drift_report = pd.DataFrame(results)

    print("\nData Drift Report")
    print("=================")
    print(drift_report.to_string(index=False))

    drift_count = drift_report["drift_detected"].sum()

    print(f"\nFeatures with drift: {drift_count}")

    if drift_count > 0:
        print("WARNING: Data drift detected!")
    else:
        print("No significant data drift detected.")


if __name__ == "__main__":
    main()