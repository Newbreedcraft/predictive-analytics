# src/model_training.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_preparation import load_and_prepare_data

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data('historical_project_data.csv')
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
