# utils.py
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pickle
import os

EMOTION_MAP = {
    0: "angry",
    1: "disgust",    # sometimes included; you may collapse to 'angry' or 'neutral'
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

def load_emotion_model(path="emotion_cnn.h5"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Train the model or place a file there.")
    return load_model(path)

def preprocess_face(img_pil, target_size=(48,48)):
    # img_pil is PIL.Image or numpy array
    if isinstance(img_pil, np.ndarray):
        img = img_pil
    else:
        img = np.array(img_pil.convert("L"))
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = img.reshape((1,target_size[0],target_size[1],1))
    return img

def predict_emotion(model, face_img):
    x = preprocess_face(face_img)
    preds = model.predict(x)
    idx = int(np.argmax(preds))
    return EMOTION_MAP.get(idx, "neutral"), float(np.max(preds))

def load_sentiment_model(path="sentiment_lstm.h5", tok_path="tokenizer.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = load_model(path)
    import pickle
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer
