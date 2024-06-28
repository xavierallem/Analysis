import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Generate synthetic time series dataset
def generate_synthetic_data(num_samples=1000, time_steps=24):  # 24 time steps (e.g., hourly data for a day)
    np.random.seed(42)

    pressure_columns = [f'pressure_{i+1}' for i in range(16)]

    data = []
    for _ in range(num_samples):
        # Generate base pressures for this sample
        base_pressures = np.random.rand(16) * 100

        for step in range(time_steps):
            # Add some random variation to base pressures for each time step
            pressures = base_pressures + np.random.randn(16) * 5
            pressures = np.clip(pressures, 0, 100)  # Ensure pressures are between 0 and 100

            sample = {f'pressure_{i+1}': pressures[i] for i in range(16)}
            sample['timestamp'] = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=step)
            sample['sample_id'] = _
            data.append(sample)

    df = pd.DataFrame(data)

    # Calculate features
    df['max_pressure'] = df[pressure_columns].max(axis=1)
    df['avg_pressure'] = df[pressure_columns].mean(axis=1)
    df['pressure_variance'] = df[pressure_columns].var(axis=1)

    return df

# Prepare data for machine learning
def prepare_data(df):
    # Group by sample_id and calculate features over time
    grouped = df.groupby('sample_id')

    features = grouped.agg({
        'max_pressure': ['max', 'mean'],
        'avg_pressure': ['max', 'mean'],
        'pressure_variance': ['max', 'mean']
    })
    features.columns = ['max_pressure_max', 'max_pressure_mean', 'avg_pressure_max', 'avg_pressure_mean', 'pressure_variance_max', 'pressure_variance_mean']

    # Calculate the number of high pressure readings for each sensor
    high_pressure_threshold = 80
    for col in [f'pressure_{i+1}' for i in range(16)]:
        features[f'{col}_high_count'] = grouped[col].apply(lambda x: (x > high_pressure_threshold).sum())

    # Define ulcer risk (this is a simplified criterion and should be adjusted based on medical expertise)
    features['ulcer_risk'] = ((features['max_pressure_max'] > 90) &
                              (features['avg_pressure_mean'] > 60) &
                              (features['pressure_variance_mean'] > 300) &
                              (features[[col for col in features.columns if '_high_count' in col]].max(axis=1) > 5)).astype(int)

    X = features.drop('ulcer_risk', axis=1)
    y = features['ulcer_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train the model and make predictions
def train_and_predict(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

# Function to predict ulcer risk for new time series data
def predict_ulcer_risk(new_data, model, scaler):
    # Prepare the new data similarly to how we prepared the training data
    pressure_columns = [f'pressure_{i+1}' for i in range(16)]

    features = pd.DataFrame({
        'max_pressure_max': new_data['max_pressure'].max(),
        'max_pressure_mean': new_data['max_pressure'].mean(),
        'avg_pressure_max': new_data['avg_pressure'].max(),
        'avg_pressure_mean': new_data['avg_pressure'].mean(),
        'pressure_variance_max': new_data['pressure_variance'].max(),
        'pressure_variance_mean': new_data['pressure_variance'].mean()
    }, index=[0])

    high_pressure_threshold = 80
    for col in pressure_columns:
        features[f'{col}_high_count'] = (new_data[col] > high_pressure_threshold).sum()

    # Scale the features
    features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

    # Make prediction
    prediction = model.predict(features_scaled)

    # Identify high-risk sensors (those with high pressure for more than 20% of the time)
    high_risk_threshold = len(new_data) * 0.2
    high_risk_sensors = [col for col in pressure_columns if features[f'{col}_high_count'].values[0] > high_risk_threshold]

    return prediction[0], high_risk_sensors

# Main execution
if __name__ == "__main__":
    # Generate synthetic time series data
    df = generate_synthetic_data()
    print("Sample data (first few rows of first sample):")
    print(df[df['sample_id'] == 0].head())

    # Prepare data for machine learning
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Train the model and make predictions
    model = train_and_predict(X_train, X_test, y_train, y_test)

    print("Model training and prediction completed.")

    # Example of new foot pressure data over time
    new_foot_pressure_data = generate_synthetic_data(num_samples=1, time_steps=24).drop('sample_id', axis=1)

    print("\nNew foot pressure data (first few rows):")
    print(new_foot_pressure_data.head())

    # Predict ulcer risk for the new data
    prediction, high_risk_sensors = predict_ulcer_risk(new_foot_pressure_data, model, scaler)

    print(f"\nPrediction for new data: {'High risk' if prediction == 1 else 'Low risk'} of ulcer")

    if high_risk_sensors:
        print("High-risk sensors detected (high pressure for >20% of the time):")
        for sensor in high_risk_sensors:
            high_pressure_count = (new_foot_pressure_data[sensor] > 80).sum()
            percentage = (high_pressure_count / len(new_foot_pressure_data)) * 100
            print(f"  - {sensor}: High pressure in {high_pressure_count} out of {len(new_foot_pressure_data)} readings ({percentage:.2f}%)")
    else:
        print("No high-risk sensors detected.")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

""" image_path = "feet.png"
image = mpimg.imread(image_path)


sensor_positions = {
    1: (640, 120), 2: (740, 130),
    3: (580, 230), 4: (670, 230), 5: (750, 230),
    6: (570, 350), 7: (660, 350), 8: (740, 350),
    9: (580, 500), 10: (650, 500), 11: (720, 500),
    12: (580, 620), 13: (640, 620), 14: (700, 620),
    15: (595, 800), 16: (675, 790)
}


results =high_risk_sensors
detected_sensors = [int(result.split('_')[1]) for result in results]


fig, ax = plt.subplots()

ax.imshow(image)

for sensor in detected_sensors:
    position = sensor_positions[sensor]
    top_left = (position[0] - 30, position[1] - 30)
    width = 60
    height = 60
    rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(position[0] - 40, position[1] - 40, f"Ulcer", fontsize=5, color='g')

ax.axis('off')


plt.show() 
 """