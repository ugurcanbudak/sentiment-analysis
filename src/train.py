from preprocess import load_data, preprocess_data, prepare_data
from model import create_model
import os
import nltk
import tensorflow as tf


nltk.download('stopwords')
nltk.download('punkt_tab')

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("Using GPU")
else:
    print("Using CPU")

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Verify the relative path
file_path = '../Sentiment Analysis/data/IMDB Dataset.csv'
if os.path.exists(file_path):
    print("File exists")
else:
    print("File does not exist")

# Use an absolute path for verification
absolute_file_path = os.path.abspath(file_path)
print("Absolute File Path:", absolute_file_path)
if os.path.exists(absolute_file_path):
    print("File exists at absolute path")
else:
    print("File does not exist at absolute path")

def train(data_path, model_path, text_column='review', label_column='sentiment', epochs=10, batch_size=32):
    df = load_data(data_path)
    df = preprocess_data(df, text_column)
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(df, text_column, label_column)

    model = create_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    model.save(model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    train(file_path, '../Sentiment Analysis/models/sentiment_model.keras')