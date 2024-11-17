import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler (using st.cache_resource)
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Thresholds for each feature to detect potential issues
thresholds = {
    'Engine rpm': 0.4,  
    'Lub oil pressure': 0.7,  
    'Fuel pressure': 0.6,  
    'Coolant pressure': 0.7,  
    'lub oil temp': 0.65,  
    'Coolant temp': 0.6,  
    'Temperature_difference': 0.65  
}

# Check if sensor values exceed the defined thresholds
def check_sensor_thresholds(data, thresholds):
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

# Predict maintenance with threshold logic and expanded time periods
def predict_maintenance_with_threshold(model, scaler, data, feature_thresholds):
    # Standardize the features
    X_scaled = scaler.transform(data)
    # Predict probabilities
    prediction_probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Check maintenance needs based on feature thresholds
    maintenance_needed = check_sensor_thresholds(data, feature_thresholds)
    
    # Combine the model prediction and sensor threshold logic
    final_maintenance_needed = [
        1 if maintenance_needed[i] == 1 or prediction_probabilities[i] > 0.5 else 0
        for i in range(len(maintenance_needed))
    ]
    
    # Expanded time periods based on probability and maintenance need
    final_time_period = []
    for i in range(len(maintenance_needed)):
        if maintenance_needed[i] == 1:
            # If maintenance is needed, check prediction probability for maintenance time
            if prediction_probabilities[i] > 0.9:
                final_time_period.append("Immediate Maintenance")
            elif prediction_probabilities[i] > 0.8:
                final_time_period.append("Maintenance needed within 7 days")
            elif prediction_probabilities[i] > 0.6:
                final_time_period.append("Maintenance needed within 14 days")
            elif prediction_probabilities[i] > 0.4:
                final_time_period.append("Maintenance needed within 30 days")
            else:
                final_time_period.append("Maintenance not required")
        else:
            # If maintenance is not needed, engine is in good condition
            final_time_period.append("Engine is in good condition")

    return final_maintenance_needed, final_time_period

# Streamlit app layout
def app():
    st.title("Vehicle Predictive Maintenance System")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the data
        st.write("Preview of the dataset:", df.head())
        
        # Load the pre-trained model and scaler
        model, scaler = load_model()

        # Ensure the columns in the uploaded file match the model's expectations
        if 'Engine rpm' in df.columns and 'Lub oil pressure' in df.columns and 'Fuel pressure' in df.columns:
            # Predict maintenance and time periods
            predictions, time_periods = predict_maintenance_with_threshold(model, scaler, df, thresholds)

            # Add predictions and time periods to the dataframe
            df['Predicted Maintenance'] = predictions
            df['Maintenance Time Period'] = time_periods

            # Display the predictions and maintenance time periods
            st.write("Predictions and Maintenance Time Periods:", df[['Predicted Maintenance', 'Maintenance Time Period']])

            # Allow the user to download the results as a CSV
            st.download_button(
                label="Download Predictions and Maintenance Times",
                data=df.to_csv(index=False),
                file_name="maintenance_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("The dataset is missing required columns like 'Engine rpm', 'Lub oil pressure', 'Fuel pressure', etc.")

if __name__ == "__main__":
    app()
