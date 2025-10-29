import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf

# Example tiny dataset (replace with real dataset like IMDB, SST, or your own labeled data)
texts = [
    "I am so happy and joyful",
    "This is terrible and sad",
    "I love this song",
    "I hate this movie",
    "Feeling excited and great",
    "I am disappointed and angry"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

def build_and_train():
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=40, padding='post')
    y = np.array(labels)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=40),
        LSTM(64),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=30, batch_size=8, verbose=1)
    model.save("models/sentiment_lstm.h5")
    # Save tokenizer
    import pickle
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Saved sentiment model and tokenizer")

if __name__ == "__main__":
    build_and_train()
