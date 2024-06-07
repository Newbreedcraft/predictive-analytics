# src/generate_synthetic_data.py

import pandas as pd
import numpy as np

def generate_synthetic_data(num_records=1000):
    np.random.seed(42)
    
    project_ids = np.arange(1, num_records + 1)
    project_names = [f'Project_{i}' for i in project_ids]
    start_dates = pd.date_range(start='2020-01-01', periods=num_records, freq='7D')
    end_dates = start_dates + pd.to_timedelta(np.random.randint(30, 100, size=num_records), unit='D')
    timelines = (end_dates - start_dates).days
    
    num_tasks = np.random.randint(10, 100, size=num_records)
    team_size = np.random.randint(5, 20, size=num_records)
    budget = np.random.randint(100000, 500000, size=num_records)
    complexity = np.random.choice(['Low', 'Medium', 'High'], size=num_records)
    resources_used = num_tasks * team_size * np.random.randint(1, 5, size=num_records)

    data = pd.DataFrame({
        'project_id': project_ids,
        'project_name': project_names,
        'start_date': start_dates,
        'end_date': end_dates,
        'timeline_days': timelines,
        'num_tasks': num_tasks,
        'team_size': team_size,
        'budget': budget,
        'complexity': complexity,
        'resources_used': resources_used
    })

    # Convert dates to strings in a readable format
    data['start_date'] = data['start_date'].dt.strftime('%Y-%m-%d')
    data['end_date'] = data['end_date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    data.to_csv('historical_project_data.csv', index=False)
    print('Synthetic data generated and saved to historical_project_data.csv')

if __name__ == '__main__':
    generate_synthetic_data()
