# 🎵 Music Genre Classification

An end-to-end Machine Learning and Deep Learning pipeline that classifies audio tracks into 10 genres using extracted acoustic features from the GTZAN dataset. Includes an interactive **Streamlit web application**, baseline ML models, a 1D Convolutional Neural Network (CNN), and Docker containerization.

---

## 📁 Project Structure

```text
Music-Genre-Classification/
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile                  # Containerization & Deployment
├── train.py                    # Step 04: Baseline Model Training (LogReg, SVM, RF)
├── train_cnn.py                # Step 05: 1D CNN Architecture & Training
├── evaluate_models.py          # Step 06: Model Evaluation, Heatmaps & Champion Selection
├── app.py                      # Step 07: Interactive Streamlit Web Interface
├── main.py                     # CLI Genre Predictor
├── notebooks/
│   └── 01_eda_and_exploration.ipynb   # Step 03: EDA (Plots, Heatmap, Missing Values)
├── models/
│   ├── champion_genre_classifier.pkl  # Champion Model (SVM)
│   ├── svm_model.pkl
│   ├── logisticregression_model.pkl
│   ├── randomforest_model.pkl
│   ├── cnn_model.keras
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── confusion_matrices.png
│   └── cnn_training_history.png
└── data/
    ├── features_30_sec.csv
    └── features_3_sec.csv
```

---

## 📊 Model Evaluation & Benchmarks

| Model | Test Accuracy | Status |
| :--- | :---: | :---: |
| **Support Vector Machine (SVM)** | **73.50%** | 🏆 **Champion Model** |
| **1D Convolutional Neural Network (CNN)** | **72.00%** (76.5% Val) | Deep Learning Model |
| **Logistic Regression** | 71.00% | Baseline Model |
| **Random Forest** | 69.50% | Baseline Model |

Confusion matrix comparisons are available at `models/confusion_matrices.png`.

---
