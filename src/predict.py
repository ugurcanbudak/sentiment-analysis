from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def predict(text, model_path, tokenizer, max_len=100):
    model = load_model(model_path)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return 'Positive' if prediction > 0.5 else 'Negative'

if __name__ == '__main__':
    from data_preprocessing import preprocess_text
    text = "I love this product!"
    preprocessed_text = preprocess_text(text)
    tokenizer = ...  # Load your tokenizer here
    print(predict(preprocessed_text, '../models/sentiment_model.h5', tokenizer))