# src/model_evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from data_preparation import load_and_prepare_data
from model_training import train_model

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data('historical_project_data.csv')
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    plot_predictions(y_test, predictions)
