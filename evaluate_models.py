import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_and_select_model(data_path="data/features_30_sec.csv", model_dir="models"):
    """
    Step 06: Model Evaluation & Selection
    - 01: Load Saved Machine Learning Models for Evaluation
    - 02: Generate Predictions with Trained Models on Test Data
    - 03: Generate Detailed Classification Reports
    - 04: Compute Confusion Matrices for Model Evaluation
    - 05: Visualize Confusion Matrices with Heatmaps
    - 06: Interpret Confusion Matrix Heatmaps for Model Comparison
    - 07: Model Performance Comparison and Selection Report
    - 08: Declare the Champion Model for Music Genre Classification
    """
    print("Loading test data for evaluation...")
    df = pd.read_csv(data_path)
    cols_to_drop = [c for c in ["filename", "length"] if c in df.columns]
    X = df.drop(columns=cols_to_drop + ["label"])
    y = df["label"]

    # -----------------------------------------------------------------
    # Step 01: Load Saved Machine Learning Models and Preprocessors
    # -----------------------------------------------------------------
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    y_encoded = label_encoder.transform(y)

    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_test_scaled = scaler.transform(X_test)

    # Dictionary of models to evaluate
    model_files = {
        "Logistic Regression": "logisticregression_model.pkl",
        "Support Vector Machine (SVM)": "svm_model.pkl",
        "Random Forest": "randomforest_model.pkl"
    }

    models = {}
    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    # -----------------------------------------------------------------
    # Step 02, 03, 04, 05: Predict, Reports, Confusion Matrices, Heatmaps
    # -----------------------------------------------------------------
    results = {}
    classes = label_encoder.classes_

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models.items()):
        # Step 02: Generate predictions
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        # Step 03: Classification report
        print(f"\n=======================================================")
        print(f" {name} - Test Accuracy: {acc * 100:.2f}%")
        print(f"=======================================================")
        print(classification_report(y_test, y_pred, target_names=classes))

        # Step 04: Compute Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # Step 05: Visualize Confusion Matrix with Heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=axes[idx],
            cbar=False
        )
        axes[idx].set_title(f"{name}\nAccuracy: {acc * 100:.2f}%", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        axes[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    cm_path = os.path.join(model_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix heatmaps saved to: {cm_path}")

    # -----------------------------------------------------------------
    # Step 06 & 07: Interpret Heatmaps and Comparison Report
    # -----------------------------------------------------------------
    print("\n================ Model Performance Comparison ================")
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f" • {model_name.ljust(30)}: {acc * 100:.2f}%")

    # -----------------------------------------------------------------
    # Step 08: Declare the Champion Model
    # -----------------------------------------------------------------
    champion_name = max(results, key=results.get)
    champion_model = models[champion_name]
    print(f"\n🏆 CHAMPION MODEL DECLARED: {champion_name} ({results[champion_name] * 100:.2f}% Accuracy)")

    # Save as the official champion model
    champion_save_path = os.path.join(model_dir, "champion_genre_classifier.pkl")
    joblib.dump(champion_model, champion_save_path)
    joblib.dump(champion_model, os.path.join(model_dir, "genre_classifier.pkl"))
    print(f"Champion model saved to: {champion_save_path}")


if __name__ == "__main__":
    evaluate_and_select_model()
