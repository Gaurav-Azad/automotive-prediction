import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tqdm import tqdm  # For the progress bar


# Thresholds for each feature to detect potential issues
thresholds = {
    'Engine rpm': 0.4,  # Lowered threshold assuming minor deviations might indicate potential issues.
    'Lub oil pressure': 0.7,  # Raised threshold to ensure early detection of lubrication system issues.
    'Fuel pressure': 0.6,  # Kept threshold unchanged assuming moderate deviations are indicative of potential problems.
    'Coolant pressure': 0.7,  # Raised threshold to detect cooling system issues more sensitively.
    'lub oil temp': 0.65,  # Slightly raised threshold to detect temperature anomalies earlier.
    'Coolant temp': 0.6,  # Kept threshold unchanged assuming moderate deviations might indicate potential issues.
    'Temperature_difference': 0.65  # Slightly raised threshold for early detection of temperature gradient anomalies.
}

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Encode categorical columns, if any
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    return df


def check_sensor_thresholds(data, thresholds):
    # Check if sensor values exceed the defined thresholds
    maintenance_needed = []
    
    for _, row in data.iterrows():
        if (row['Engine rpm'] > thresholds['Engine rpm'] or 
            row['Lub oil pressure'] > thresholds['Lub oil pressure'] or 
            row['Fuel pressure'] > thresholds['Fuel pressure'] or 
            row['Coolant pressure'] > thresholds['Coolant pressure'] or
            row['lub oil temp'] > thresholds['lub oil temp'] or 
            row['Coolant temp'] > thresholds['Coolant temp'] or 
            row['Temperature_difference'] > thresholds['Temperature_difference']):
            maintenance_needed.append(1)  # Maintenance needed
        else:
            maintenance_needed.append(0)  # No maintenance needed
    
    return maintenance_needed


def train_model(data, target_column):
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    print("Standardizing features...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting Classifier with predefined parameters
    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,  # Specify the number of trees
        learning_rate=0.05,  # Learning rate
        max_depth=6,  # Maximum depth of each tree
        random_state=42  # For reproducibility
    )
    
    # Add progress bar during training
    with tqdm(total=1, desc="Training Progress") as progress_bar:
        model.fit(X_train_scaled, y_train)
        progress_bar.update(1)
    
    # Evaluate model
    print("Evaluating the model...")
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model and scaler saved successfully.")


# Predict maintenance with threshold logic and expanded time periods
def predict_maintenance_with_threshold(model, scaler, data, feature_thresholds):
    # Standardize the features
    X_scaled = scaler.transform(data)
    # Predict probabilities
    prediction_probabilities = model.predict_proba(X_scaled)[:, 1]
    print(prediction_probabilities)
    
    # Check maintenance needs based on feature thresholds
    maintenance_needed = check_sensor_thresholds(data, feature_thresholds)
    # Combine the model prediction and sensor threshold logic
    final_maintenance_needed = [
        1 if maintenance_needed[i] == 1 or prediction_probabilities[i] > 0.5 else 0
        for i in range(len(maintenance_needed))
    ]
    
    # Expanded time periods based on probability and maintenance need
    final_time_period = [
        "Immediate Maintenance" if maintenance_needed[i] == 1 or prediction_probabilities[i] > 0.9 else
        "Maintenance needed within 7 days" if maintenance_needed[i] == 1 or prediction_probabilities[i] > 0.8 else
        "Maintenance needed within 14 days" if maintenance_needed[i] == 1 or prediction_probabilities[i] > 0.6 else
        "Maintenance needed within 30 days" if maintenance_needed[i] == 1 or prediction_probabilities[i] > 0.4 else
        "Maintenance not required"
        for i in range(len(maintenance_needed))
    ]
    
    return final_maintenance_needed, final_time_period


if __name__ == "__main__":
    # File path to your dataset
    dataset_path = "ML-Based-Vehicle-Predictive-Maintenance-System-with-Real-Time-Visualization-main/engine_data.csv"  # Replace with your dataset's file path
    df = load_and_preprocess_data(dataset_path)
    target_col = "Engine Condition"  # Replace with the actual target column name in your dataset
    
    # Train the model
    train_model(df, target_col)
    
    # Load model and scaler
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    
    # Make predictions on new data
    predictions, time_periods = predict_maintenance_with_threshold(model, scaler, df.drop(columns=[target_col]), thresholds)
    
    # Add the predictions and time periods to the dataframe
    df['Predicted Maintenance'] = predictions
    df['Maintenance Time Period'] = time_periods
    
    # Save results to CSV
    df.to_csv("engine_maintenance_predictions.csv", index=False)
    print("Predictions and maintenance time periods saved to 'engine_maintenance_predictions.csv'.")
