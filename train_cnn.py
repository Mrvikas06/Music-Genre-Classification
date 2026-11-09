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

    # -------------------------------------------------------------
    # Step 11: Compile the Convolutional Neural Network
    # -------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # -------------------------------------------------------------
    # Step 12: Train the CNN for Music Genre Classification
    # -------------------------------------------------------------
    print("\n--- Training 1D CNN Model ---")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    ]
    history = model.fit(
        X_train_cnn,
        y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc * 100:.2f}%")

    y_pred_probs = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("\nClassification Report for 1D CNN:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # -------------------------------------------------------------
    # Step 13: Visualize Model Training History with Matplotlib
    # -------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy Over Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax2.set_title("Model Loss Over Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    history_plot_path = os.path.join(output_dir, "cnn_training_history.png")
    plt.savefig(history_plot_path)
    plt.close()
    print(f"Training history visualization saved to: {history_plot_path}")

    # -------------------------------------------------------------
    # Step 14: Save a Trained Keras CNN Model
    # -------------------------------------------------------------
    model_save_path = os.path.join(output_dir, "cnn_model.keras")
    model.save(model_save_path)
    print(f"Trained Keras CNN model saved to: {model_save_path}")


if __name__ == "__main__":
    train_cnn_model()
