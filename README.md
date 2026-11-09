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

## 🚀 Quick Start

### 1. Installation
```bash
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux

pip install -r requirements.txt
```

### 2. Train Models & Evaluate
```bash
# Train baseline models
python train.py

# Train 1D CNN
python train_cnn.py

# Run comprehensive evaluation and generate heatmaps
python evaluate_models.py
```

### 3. Launch Interactive Web App (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, upload any `.wav` or `.mp3` audio track, and view the predicted genre with confidence breakdown.

---

## 🐳 Docker Deployment

### 1. Build Docker Image
```bash
docker build -t music-genre-classifier .
```

### 2. Run Docker Container
```bash
docker run -p 8501:8501 music-genre-classifier
```
Access the application at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud
1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "Complete ML pipeline, EDA, CNN, Streamlit app, and Docker setup"
   git push origin main
   ```
2. Log into [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, set the main file path to `app.py`, and click **Deploy**!
