import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data(train_path="train.csv", test_path="test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("=== DATA LOADED ===")
    print("Train shape:", train.shape)
    print("Test shape: ", test.shape)

    print("\n=== TRAIN HEAD ===")
    print(train.head())

    print("\n=== TRAIN INFO ===")
    print(train.info())

    return train, test


# =========================================================
# 2. CLEANING (MINIMAL FOR DIGIT DATA)
# =========================================================
def basic_cleaning(train, test, target_col="label"):
    before = train.shape[0]
    train = train.drop_duplicates()
    after = train.shape[0]
    print(f"\nRemoved {before - after} duplicate rows from train")

    train["is_train"] = 1
    test["is_train"] = 0

    full = pd.concat([train, test], ignore_index=True)

    numeric_cols = full.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    for col in numeric_cols:
        full[col] = full[col].fillna(full[col].median())

    print("Missing values handled.")
    return full


# =========================================================
# 3. SPLIT BACK INTO X, y, X_test
# =========================================================
def split_full_to_data(full, target_col="label"):
    train_clean = full[full["is_train"] == 1].copy()
    test_clean = full[full["is_train"] == 0].copy()

    train_clean = train_clean.drop(columns=["is_train"])
    test_clean = test_clean.drop(columns=["is_train"])

    y = train_clean[target_col]
    X = train_clean.drop(columns=[target_col], errors="ignore")

    X_test = test_clean.drop(columns=[target_col], errors="ignore")

    return X, y, X_test


# =========================================================
# 4. TRAIN + VALIDATE (80/20)
# =========================================================
def train_and_validate_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Val shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining RandomForest...")
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {acc:.4f}")

    return model


# =========================================================
# 5. TRAIN ON FULL DATA + PREDICT TEST
# =========================================================
def train_on_full_and_predict(model, X, y, X_test):
    print("\nTraining model on FULL data...")
    model.fit(X, y)

    print("Creating test predictions...")
    test_preds = model.predict(X_test)

    return test_preds


# =========================================================
# 6. CREATE SUBMISSION FILE
# =========================================================
def create_submission(test_preds, filename="submission_NEW.csv"):
    import numpy as np
    # make sure integer labels (no .0)
    test_preds = np.asarray(test_preds).astype(int)

    submission = pd.DataFrame({
        "ImageId": range(1, len(test_preds) + 1),
        "Label": test_preds
    })

    # final sanity checks
    assert submission.shape[0] > 0
    assert submission.columns.tolist() == ["ImageId", "Label"]
    assert submission["Label"].isnull().sum() == 0
    assert submission["ImageId"].min() == 1
    assert submission["ImageId"].max() == len(test_preds)

    submission.to_csv(filename, index=False)
    print(f"Saved: {filename}")
    print(submission.head())



# =========================================================
# MAIN
# =========================================================
def main():
    train, test = load_data("train.csv", "test.csv")

    full = basic_cleaning(train, test, target_col="label")

    X, y, X_test = split_full_to_data(full, target_col="label")

    print("\nShapes after split:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("X_test:", X_test.shape)

    model = train_and_validate_model(X, y)

    test_preds = train_on_full_and_predict(model, X, y, X_test)

    # after you get test_preds
    print("Test preds shape:", test_preds.shape)
    print("Unique preds:", np.unique(test_preds, return_counts=True))
    print("X_test shape:", X_test.shape)
    print("X columns (count):", X.shape[1], "X_test columns (count):", X_test.shape[1])
    print("Submission head:\n", pd.DataFrame({
        "ImageId": range(1, len(test_preds)+1),
        "Label": test_preds
    }).head())


    create_submission(test_preds, filename="Submission_real_new.csv")


if __name__ == "__main__":
    main()
