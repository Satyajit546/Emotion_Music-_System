# app.py
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import os
import nltk
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from utils import load_emotion_model, predict_emotion, load_sentiment_model
from spotify_utils import search_tracks_by_mood

nltk.download('punkt')

st.set_page_config(layout="centered", page_title="🎵 Emotion-based Music Recommender")
st.title("🎵 Emotion-based Music Recommendation System")

# ---------------- MODEL LOADERS ----------------
@st.cache_resource
def get_emotion_model():
    return load_emotion_model("emotion_cnn.h5")

@st.cache_resource
def get_sentiment():
    try:
        m, t = load_sentiment_model("sentiment_lstm.h5", "tokenizer.pkl")
        return m, t
    except Exception:
        return None, None

emotion_model = get_emotion_model()
sent_model, tokenizer = get_sentiment()

# ---------------- APP TABS ----------------
tab1, tab2, tab3 = st.tabs(["🖼️ Emotion (Image)", "💬 Text Sentiment", "🎥 Live Camera Emotion Music"])

# ========== TAB 1 : EMOTION FROM IMAGE ==========
with tab1:
    st.header("Detect emotion from image")
    source = st.radio("Image source", ["Upload image", "Use camera"])
    img = None
    if source == "Upload image":
        uploaded = st.file_uploader("Upload a face image", type=["png","jpg","jpeg"])
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)
    else:
        cam = st.camera_input("Take a photo")
        if cam is not None:
            bytes_data = cam.getvalue()
            img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            st.image(img, caption="Captured image", use_column_width=True)

    if img is not None:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            st.warning("No face detected. Try a clearer, closer image.")
        else:
            x, y, w, h = faces[0]
            face_img = Image.fromarray(gray[y:y + h, x:x + w])
            emotion, conf = predict_emotion(emotion_model, face_img)
            st.success(f"Detected emotion: **{emotion}** (confidence {conf:.2f})")

            st.subheader("🎶 Recommended Tracks")
            try:
                tracks = search_tracks_by_mood(emotion, limit=8)
                for t in tracks:
                    st.markdown(f"**{t['name']}** — {t['artist']}")
                    if t['preview_url']:
                        st.audio(t['preview_url'])
                    st.markdown(f"[Open in Spotify]({t['spotify_url']})")
            except Exception:
                st.error("Spotify not available. Using local fallback.")
                fallback = {
                    "happy": ["Happy Song 1 - Artist", "Happy Song 2 - Artist"],
                    "sad": ["Sad Song 1 - Artist", "Sad Song 2 - Artist"],
                    "angry": ["Angry Song 1 - Artist"],
                    "neutral": ["Chill Song 1 - Artist"]
                }
                for s in fallback.get(emotion.lower(), ["No local songs available."]):
                    st.write("- " + s)

# ========== TAB 2 : SENTIMENT FROM TEXT ==========
with tab2:
    st.header("Analyze text sentiment")
    txt = st.text_area("Enter text (lyrics, message, or review):", height=120)
    if st.button("Analyze Sentiment"):
        if not txt.strip():
            st.warning("Please enter text first.")
        else:
            if sent_model is not None and tokenizer is not None:
                seq = tokenizer.texts_to_sequences([txt])
                seq = pad_sequences(seq, maxlen=40, padding='post')
                score = float(sent_model.predict(seq)[0, 0])
                sentiment = "positive" if score > 0.5 else "negative"
                st.success(f"Sentiment: **{sentiment}** (score {score:.2f})")
            else:
                pos = {"love","happy","joy","excited","good","amazing","great"}
                neg = {"hate","sad","terrible","angry","disappointed","bad"}
                words = set(nltk.word_tokenize(txt.lower()))
                p = len(words & pos)
                n = len(words & neg)
                sentiment = "positive" if p >= n else "negative"
                st.info(f"(Fallback) Sentiment: **{sentiment}** — pos={p}, neg={n}")

            mood = "happy" if sentiment == "positive" else "sad"
            try:
                tracks = search_tracks_by_mood(mood, limit=6)
                st.subheader("🎧 Recommended Tracks")
                for t in tracks:
                    st.markdown(f"**{t['name']}** — {t['artist']}")
                    if t['preview_url']:
                        st.audio(t['preview_url'])
                    st.markdown(f"[Open in Spotify]({t['spotify_url']})")
            except Exception:
                st.warning("Spotify API not available; try again later.")

# ========== TAB 3 : LIVE CAMERA EMOTION MUSIC ==========
with tab3:
    st.header("🎥 Live Emotion Detection & Music")
    st.markdown("Detect emotions (Happy 😄, Sad 😢, Angry 😡) from webcam or IP camera and auto-play matching songs!")

    # Emotion label mapping (adjust based on your model output)
    EMOTION_LABELS = {0: "Angry", 3: "Happy", 4: "Sad"}
    SONG_MAP = {
        "Happy": "songs/happy.mp3",
        "Sad": "songs/sad.mp3",
        "Angry": "songs/angry.mp3"
    }

    # Camera source input
    CAMERA_URL = st.text_input("Enter camera URL (leave blank for default webcam):", "")
    if CAMERA_URL.strip():
        camera = cv2.VideoCapture(CAMERA_URL)
        st.write(f"Using external camera stream: `{CAMERA_URL}`")
    else:
        camera = cv2.VideoCapture(0)
        st.write("Using default webcam.")

    # Check connection
    if not camera.isOpened():
        st.error("❌ Cannot open camera stream. Check URL or connection.")
    else:
        st.success("✅ Camera connected successfully!")
        run = st.checkbox("Start Live Detection")
        FRAME_WINDOW = st.image([])
        last_emotion = None

        if run:
            st.info("Camera started. Keep your face visible and well-lit.")
            while True:
                ret, frame = camera.read()
                if not ret:
                    st.error("Could not access webcam stream.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
                            .detectMultiScale(gray, 1.3, 5)
                current_emotion = None

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi = roi_gray / 255.0
                    roi = np.reshape(roi, (1, 48, 48, 1))
                    preds = emotion_model.predict(roi)
                    label_idx = int(np.argmax(preds))
                    confidence = float(np.max(preds))

                    if label_idx in EMOTION_LABELS:
                        emotion = EMOTION_LABELS[label_idx]
                        current_emotion = emotion
                        color = (0, 255, 0) if emotion == "Happy" else (255, 0, 0) if emotion == "Sad" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

                # 🎵 Play song when emotion changes
                if current_emotion and current_emotion != last_emotion:
                    last_emotion = current_emotion
                    st.subheader(f"🎶 Current Emotion: {current_emotion}")
                    song_path = SONG_MAP.get(current_emotion)
                    if song_path and os.path.exists(song_path):
                        st.audio(song_path, format='audio/mp3')
                    else:
                        st.warning(f"No local song found for {current_emotion} emotion.")

                if not run:
                    break

        camera.release()
        st.write("Camera stopped.")

st.markdown("---")
st.caption("Integrated Emotion, Sentiment, and Live Music modules for demonstration.")
