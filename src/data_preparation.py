# src/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath, parse_dates=['start_date', 'end_date'])
    data['project_duration'] = (data['end_date'] - data['start_date']).dt.days
    data['complexity_encoded'] = LabelEncoder().fit_transform(data['complexity'])
    
    features = data[['num_tasks', 'team_size', 'budget', 'complexity_encoded', 'resources_used']]
    target = data['project_duration']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data('historical_project_data.csv')
    print("Data prepared successfully.")
