import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "outputs" / "landmark_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LANDMARK DATASET SPLITTING")
    print("=" * 60)

    print(f"\nInput:")
    print(INPUT_FILE)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print("\nERROR: landmark_dataset.csv not found.")

        print(
            "\nRun this first:"
        )

        print(
            "python src/extract_landmarks.py"
        )

        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading landmark dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Total samples: {len(df)}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if "label" not in df.columns:

        print(
            "\nERROR: 'label' column not found."
        )

        return

    feature_columns = [
        column
        for column in df.columns
        if column.startswith(("x", "y", "z"))
    ]

    print(
        f"Feature columns: {len(feature_columns)}"
    )

    if len(feature_columns) != 63:

        print(
            "\nWARNING: Expected 63 landmark features."
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution before split:")
    print("-" * 60)

    class_counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in class_counts.items():

        print(
            f"{label:>2}: {count:3d}"
        )

    print(
        f"\nNumber of classes: {df['label'].nunique()}"
    )

    # --------------------------------------------------------
    # First split:
    #
    # 70% training
    # 30% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(

        df,

        train_size=TRAIN_SIZE,

        stratify=df["label"],

        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Second split:
    #
    # Temporary 30%
    #
    # 50% validation
    # 50% test
    #
    # Therefore:
    #
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(

        temp_df,

        test_size=0.5,

        stratify=temp_df["label"],

        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Reset indexes
    # --------------------------------------------------------

    train_df = train_df.reset_index(drop=True)

    validation_df = validation_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    train_file = OUTPUT_DIR / "train.csv"

    validation_file = OUTPUT_DIR / "validation.csv"

    test_file = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_file,
        index=False
    )

    validation_df.to_csv(
        validation_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    # --------------------------------------------------------
    # Overall split summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)

    print(
        f"\nTraining samples:   {len(train_df)}"
    )

    print(
        f"Validation samples: {len(validation_df)}"
    )

    print(
        f"Test samples:       {len(test_df)}"
    )

    print(
        f"Total samples:      "
        f"{len(train_df) + len(validation_df) + len(test_df)}"
    )

    # --------------------------------------------------------
    # Verify every class exists in every split
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CLASS DISTRIBUTION AFTER SPLIT")
    print("=" * 60)

    train_counts = (
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    validation_counts = (
        validation_df["label"]
        .value_counts()
        .sort_index()
    )

    test_counts = (
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\n{'Class':<10}"
        f"{'Train':>8}"
        f"{'Validation':>12}"
        f"{'Test':>8}"
    )

    print("-" * 40)

    all_classes = sorted(
        df["label"].unique()
    )

    for label in all_classes:

        train_count = train_counts.get(
            label,
            0
        )

        validation_count = validation_counts.get(
            label,
            0
        )

        test_count = test_counts.get(
            label,
            0
        )

        print(
            f"{label:<10}"
            f"{train_count:>8}"
            f"{validation_count:>12}"
            f"{test_count:>8}"
        )

    # --------------------------------------------------------
    # Check for missing classes
    # --------------------------------------------------------

    missing_train = set(all_classes) - set(
        train_df["label"].unique()
    )

    missing_validation = set(all_classes) - set(
        validation_df["label"].unique()
    )

    missing_test = set(all_classes) - set(
        test_df["label"].unique()
    )

    print("\n")
    print("=" * 60)
    print("SPLIT VALIDATION")
    print("=" * 60)

    if not missing_train:
        print("✓ All classes present in training set")
    else:
        print(
            f"✗ Missing from training: {missing_train}"
        )

    if not missing_validation:
        print("✓ All classes present in validation set")
    else:
        print(
            f"✗ Missing from validation: "
            f"{missing_validation}"
        )

    if not missing_test:
        print("✓ All classes present in test set")
    else:
        print(
            f"✗ Missing from test: {missing_test}"
        )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FILES CREATED")
    print("=" * 60)

    print(
        f"\n1. {train_file}"
    )

    print(
        f"2. {validation_file}"
    )

    print(
        f"3. {test_file}"
    )

    print("\nNext step:")
    print(
        "Inspect the split before training."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    