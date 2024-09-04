import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logging.info(f"Data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}.")
        raise

def preprocess_data(data):
    """Convert JSON data to a DataFrame and preprocess it."""
    df = pd.json_normalize(data)
    
    # Convert time columns to datetime
    try:
        df['preferred_time'] = pd.to_datetime(df['preferred_time'], format='%H:%M').dt.time
        df['delivery_time'] = pd.to_datetime(df['delivery_time'], format='%H:%M').dt.time
        logging.info("Time columns successfully converted to datetime.")
    except ValueError:
        logging.error("Time format should be HH:MM")
        raise
    
    # Convert times to minutes since midnight
    df['preferred_time_minutes'] = df['preferred_time'].apply(lambda x: x.hour * 60 + x.minute)
    df['delivery_time_minutes'] = df['delivery_time'].apply(lambda x: x.hour * 60 + x.minute)
    logging.info("Time columns successfully converted to minutes since midnight.")
    
    return df

def train_model(df):
    """Train a machine learning model."""
    if df.empty:
        logging.error("DataFrame is empty. No data to train the model.")
        raise ValueError("DataFrame is empty. No data to train the model.")
    
    required_columns = ['preferred_time_minutes', 'delivery_time_minutes']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"DataFrame must contain the following columns: {required_columns}")
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    X = df[['preferred_time_minutes']]
    y = df['delivery_time_minutes']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f'Mean Squared Error: {mse:.2f}')
    
    return model

def save_model(model, file_path):
    """Save the trained model to a file using joblib."""
    try:
        joblib.dump(model, file_path)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        raise

def main():
    # Paths
    json_file = 'delivery.json'
    model_file = 'trained_model.pkl'
    
    try:
        # Load and preprocess data
        data = load_data(json_file)
        df = preprocess_data(data)
        
        # Train and save the model
        model = train_model(df)
        save_model(model, model_file)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
