import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_baseline_models(data_path="data/features_30_sec.csv", output_dir="models"):
    """
    Step 04: Training Baseline Models
    - 01: Separate Features (X) and Target (y)
    - 02: Encode Target Labels for Machine Learning
    - 03: Split Dataset into Training and Testing Sets
    - 04: Fit StandardScaler on Training Data
    - 05: Apply StandardScaler Transformation to Datasets
    - 06: Train a Baseline Logistic Regression Model
    - 07: Train a Support Vector Machine (SVM) Classifier
    - 08: Train a Random Forest Classifier
    - 09: Evaluate Trained Models and Calculate Accuracy
    - 10: Save Trained Machine Learning Models and Scaler
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature dataset not found at {data_path}")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # ---------------------------------------------------------
    # 01. Separate Features (X) and Target (y) for Modeling
    # ---------------------------------------------------------
    cols_to_drop = [c for c in ["filename", "length"] if c in df.columns]
    X = df.drop(columns=cols_to_drop + ["label"])
    y = df["label"]
    print(f"Features shape: {X.shape}, Target count: {len(y)}")

    # ---------------------------------------------------------
    # 02. Encode Target Labels for Machine Learning
    # ---------------------------------------------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Encoded {len(label_encoder.classes_)} unique classes: {list(label_encoder.classes_)}")

    # ---------------------------------------------------------
    # 03. Split Dataset into Training and Testing Sets
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # ---------------------------------------------------------
    # 04. Fit StandardScaler on Training Data
    # ---------------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(X_train)

    # ---------------------------------------------------------
    # 05. Apply StandardScaler Transformation to Datasets
    # ---------------------------------------------------------
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dictionary to hold models and evaluation scores
    models = {}
    scores = {}

    # ---------------------------------------------------------
    # 06. Train a Baseline Logistic Regression Model
    # ---------------------------------------------------------
    print("\n--- Training Baseline Logistic Regression ---")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    models["LogisticRegression"] = log_reg

    # ---------------------------------------------------------
    # 07. Train a Support Vector Machine (SVM) Classifier
