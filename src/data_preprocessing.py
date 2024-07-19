import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(url):
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    data = pd.read_csv(url, names=column_names, na_values="?")
    return data

def preprocess_data(data):
    # Handling missing values
    data = data.dropna()

    # Splitting features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    data = load_data(url)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Saving the processed data
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    X_test.to_csv('../data/processed/X_test.csv', index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv', index=False)
