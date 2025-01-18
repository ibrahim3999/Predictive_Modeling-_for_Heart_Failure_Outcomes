from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AdvancedNNModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.y_pred = None

        print("Normalizing data...")
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        print("Data normalization completed.")

    def build_model(self):
        print("Building the binary neural network model...")

        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            BatchNormalization(),
            Dropout(0.15),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.15),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.15),
            Dense(1, activation='sigmoid')  # Sigmoid for binary classification
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',  # Binary crossentropy for binary classification
                           metrics=['accuracy'])
        print("Model built successfully.")

    def train_model(self, epochs=50, batch_size=32):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Model summary:")
        self.model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        def scheduler(epoch, lr):
            return lr if epoch < 10 else lr * tf.math.exp(-0.1)

        lr_scheduler = LearningRateScheduler(scheduler)

        print("Starting model training...")
        self.model.fit(self.X_train_scaled, self.y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(self.X_test_scaled, self.y_test),
                       callbacks=[early_stopping, lr_scheduler],
                       verbose=1)
        print("Model training completed.")

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("Evaluating the model...")
        loss, accuracy = self.model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        print(f"Model Evaluation: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

        self.y_pred = self.model.predict(self.X_test_scaled)
        y_pred_classes = (self.y_pred > 0.5).astype("int32")  # Convert probabilities to binary classes

        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_classes, zero_division=0))
