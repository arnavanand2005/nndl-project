import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# FILES
# ============================================================

FILES = {
    "TRAIN": OUTPUT_DIR / "train.csv",
    "VALIDATION": OUTPUT_DIR / "validation.csv",
    "TEST": OUTPUT_DIR / "test.csv"
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LANDMARK DATASET INSPECTION")
    print("=" * 60)

    for split_name, file_path in FILES.items():

        print("\n" + "=" * 60)
        print(split_name)
        print("=" * 60)

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        df = pd.read_csv(file_path)

        print(
            f"\nSamples: {len(df)}"
        )

        # ----------------------------------------------------
        # Identify features
        # ----------------------------------------------------

        feature_columns = [
            column
            for column in df.columns
            if column.startswith(("x", "y", "z"))
        ]

        X = df[feature_columns]

        y = df["label"]

        print(
            f"Features: {len(feature_columns)}"
        )

        print(
            f"Classes: {y.nunique()}"
        )

        # ----------------------------------------------------
        # Check missing values
        # ----------------------------------------------------

        missing_values = X.isna().sum().sum()

        print(
            f"\nMissing values: {missing_values}"
        )

        # ----------------------------------------------------
        # Check infinite values
        # ----------------------------------------------------

        infinite_values = np.isinf(
            X.to_numpy()
        ).sum()

        print(
            f"Infinite values: {infinite_values}"
        )

        # ----------------------------------------------------
        # Numeric range
        # ----------------------------------------------------

        print("\nFeature statistics:")

        print(
            f"Minimum: {X.min().min():.6f}"
        )

        print(
            f"Maximum: {X.max().max():.6f}"
        )

        print(
            f"Mean:    {X.mean().mean():.6f}"
        )

        print(
            f"Std:     {X.stack().std():.6f}"
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        print("\nLabels:")

        labels = sorted(
            y.unique()
        )

        print(
            labels
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        valid = True

        if len(feature_columns) != 63:

            print(
                "\n✗ Expected 63 features."
            )

            valid = False

        if missing_values > 0:

            print(
                "\n✗ Missing values detected."
            )

            valid = False

        if infinite_values > 0:

            print(
                "\n✗ Infinite values detected."
            )

            valid = False

        if y.nunique() != 24:

            print(
                "\n✗ Expected 24 classes."
            )

            valid = False

        if valid:

            print(
                "\n✓ Dataset looks valid."
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

    print(
        "\nIf all three splits show ✓ Dataset looks valid,"
    )

    print(
        "we can proceed to model training."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()