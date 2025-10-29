# fer_train.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

CSV_PATH = "fer2013.csv"  # adjust if different
IMG_SIZE = 48
NUM_CLASSES = 7  # common FER labels: 0..6

def load_fer(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    # CSV columns typically: emotion, pixels, Usage
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].astype(int).tolist()
    X = np.array([np.fromstring(p, sep=' ', dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE) for p in pixels])
    X = X[..., np.newaxis] / 255.0
    y = to_categorical(emotions, NUM_CLASSES)
    return X, y

def build_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y = load_fer()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    model = build_model(input_shape=(IMG_SIZE,IMG_SIZE,1), num_classes=y.shape[1])
    model.summary()
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val,y_val))
    model.save("models/emotion_cnn.h5")
    print("Saved models/emotion_cnn.h5")

if __name__ == "__main__":
    main()
