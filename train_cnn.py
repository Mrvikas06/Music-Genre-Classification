import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set Keras backend to PyTorch (compatible with Python 3.14)
os.environ["KERAS_BACKEND"] = "torch"

# Step 02: Import Keras layers and components
import keras
from keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


def train_cnn_model(data_path="data/features_30_sec.csv", output_dir="models"):
    """
    Step 05: Developing a Neural Network Model (1D CNN)
    - 01: Reshape Data for 1D CNN Input
    - 02: Import Keras Layers for a CNN Model
    - 03: Initialize a Keras Sequential Model
    - 04: Add a 1D Convolutional Layer to a Sequential Model
    - 05: Add a MaxPooling1D Layer to the CNN
    - 06: Add a BatchNormalization Layer to a CNN
    - 07: Deepen the CNN with Additional Layers
    - 08: Prepare for Classification: Add a Flatten Layer
    - 09: Add a Dense Hidden Layer with Dropout to a CNN
    - 10: Finalize the CNN Architecture with an Output Layer
    - 11: Compile the Convolutional Neural Network
    - 12: Train the CNN for Music Genre Classification
    - 13: Visualize Model Training History with Matplotlib
    - 14: Save a Trained Keras CNN Model
    """
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)

    cols_to_drop = [c for c in ["filename", "length"] if c in df.columns]
    X = df.drop(columns=cols_to_drop + ["label"]).values
    y = df["label"].values

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------------------------------------
    # Step 01: Reshape Data for 1D CNN Input (samples, time_steps/features, channels)
    # -------------------------------------------------------------
    # CNN 1D expects shape: (batch_size, num_features, 1)
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)
    input_shape = (X_train_cnn.shape[1], 1)
    print(f"Reshaped Input for 1D CNN: {X_train_cnn.shape}, Input shape: {input_shape}")

    # -------------------------------------------------------------
    # Step 03: Initialize a Keras Sequential Model
    # -------------------------------------------------------------
    model = keras.Sequential()

    # Step 04: Add a 1D Convolutional Layer
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same", input_shape=input_shape))

    # Step 05: Add a MaxPooling1D Layer
    model.add(layers.MaxPooling1D(pool_size=2))

    # Step 06: Add a BatchNormalization Layer
    model.add(layers.BatchNormalization())

    # Step 07: Deepen the CNN with Additional Layers
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding="same"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())

    # Step 08: Prepare for Classification: Add a Flatten Layer
    model.add(layers.Flatten())

    # Step 09: Add a Dense Hidden Layer with Dropout to a CNN
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4))

    # Step 10: Finalize the CNN Architecture with an Output Layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.summary()
