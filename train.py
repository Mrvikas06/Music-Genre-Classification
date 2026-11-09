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
    # ---------------------------------------------------------
    print("--- Training Support Vector Machine (SVM) ---")
    svm_clf = SVC(kernel="rbf", probability=True, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    models["SVM"] = svm_clf

    # ---------------------------------------------------------
    # 08. Train a Random Forest Classifier
    # ---------------------------------------------------------
    print("--- Training Random Forest Classifier ---")
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train)
    models["RandomForest"] = rf_clf

    # ---------------------------------------------------------
    # 09. Evaluate Trained Models and Calculate Accuracy
    # ---------------------------------------------------------
    print("\n================== Model Evaluation ==================")
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        scores[name] = acc
        print(f"\n{name} Test Accuracy: {acc * 100:.2f}%")
        print(f"Classification Report for {name}:\n")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    best_model_name = max(scores, key=scores.get)
    print(f"Best Performing Model: {best_model_name} ({scores[best_model_name] * 100:.2f}%)")

    # ---------------------------------------------------------
    # 10. Save Trained Machine Learning Models and Scaler
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    joblib.dump(list(X.columns), os.path.join(output_dir, "feature_names.pkl"))

    # Save all models as well as the standard genre_classifier.pkl (pointing to best model)
    for name, model in models.items():
        model_filename = os.path.join(output_dir, f"{name.lower()}_model.pkl")
        joblib.dump(model, model_filename)
        print(f"Saved: {model_filename}")

    # Standard primary classifier used by main.py
    joblib.dump(models[best_model_name], os.path.join(output_dir, "genre_classifier.pkl"))
    print(f"Primary classifier saved to: {os.path.join(output_dir, 'genre_classifier.pkl')}")
    print(f"\nAll models, scaler, and encoders successfully saved to '{output_dir}'.")


if __name__ == "__main__":
    train_baseline_models()
