from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf  # וודא שזה קיים בראש הקובץ
import numpy as np


from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import tensorflow as tf  # וודא שזה קיים בראש הקובץ
import numpy as np
from sklearn.metrics import classification_report


class BasicNNModel:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    def build_model(self, input_dim):
        print("Building the model...")

        self.model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        print("Model built successfully.")

    def train_model(self, epochs=50, batch_size=32):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")


        print("Training the model...")
        self.model.fit(self.X_train_scaled, self.y_train,
                       epochs=epochs, batch_size=batch_size,
                       validation_data=(self.X_test_scaled, self.y_test),
                       )
        print("Model trained successfully.")

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        #print("Evaluating the model...")
        results = self.model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        print(f"Model Evaluation Results: Loss = {results[0]:.4f}, Accuracy = {results[1]:.4f}")

        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_classes = (y_pred > 0.5).astype("int32")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_classes, zero_division=0))
