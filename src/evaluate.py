from data_preprocessing import load_data, preprocess_data, prepare_data
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

def evaluate(data_path, model_path, text_column='review', label_column='sentiment'):
    df = load_data(data_path)
    df = preprocess_data(df, text_column)
    _, X_test, _, y_test, _ = prepare_data(df, text_column, label_column)

    model = load_model(model_path)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    evaluate('../data/IMDB Dataset.csv', '../models/sentiment_model.h5')