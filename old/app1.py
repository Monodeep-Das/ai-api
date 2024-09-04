from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pickle
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load the trained model
try:
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logging.info("Trained model loaded successfully.")
    logging.info(f"Model type: {type(model)}")  # Debugging line to check model type
except FileNotFoundError:
    logging.error("Trained model file not found.")
    raise RuntimeError("Trained model file not found.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Error loading model: {str(e)}")

# Define the data model
class DeliveryData(BaseModel):
    preferred_time: str  # Assuming HH:MM format
    delivery_time: str   # Assuming HH:MM format

@app.post("/process-data/")
async def process_data(data: List[DeliveryData]):
    try:
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")

        # Convert list of DeliveryData to DataFrame
        df = pd.DataFrame([item.dict() for item in data])
        
        # Convert time columns to datetime
        try:
            df['preferred_time'] = pd.to_datetime(df['preferred_time'], format='%H:%M').dt.time
            df['delivery_time'] = pd.to_datetime(df['delivery_time'], format='%H:%M').dt.time
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid time format, should be HH:MM")

        # Convert time columns to minutes since midnight for analysis and prediction
        df['preferred_time_minutes'] = df['preferred_time'].apply(lambda x: x.hour * 60 + x.minute)
        df['delivery_time_minutes'] = df['delivery_time'].apply(lambda x: x.hour * 60 + x.minute)
        
        avg_preferred_time = df['preferred_time_minutes'].mean()
        avg_delivery_time = df['delivery_time_minutes'].mean()

        if pd.isna(avg_preferred_time):
            avg_preferred_time = 0
        if pd.isna(avg_delivery_time):
            avg_delivery_time = 0

        analysis_result = {
            'average_preferred_time': f'{int(avg_preferred_time // 60)}:{int(avg_preferred_time % 60):02d}',
            'average_delivery_time': f'{int(avg_delivery_time // 60)}:{int(avg_delivery_time % 60):02d}'
        }
        
        # Make predictions using the trained model
        X = df[['preferred_time_minutes']]
        try:
            predictions = model.predict(X)
        except AttributeError:
            raise HTTPException(status_code=500, detail="Model does not have a predict method.")
        
        # Prepare prediction results
        prediction_results = list(predictions)

        # Create and save visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df['preferred_time_minutes'],
            df['delivery_time_minutes'],
            alpha=0.5
        )
        plt.title('Preferred Time vs Delivery Time')
        plt.xlabel('Preferred Time (Minutes since Midnight)')
        plt.ylabel('Delivery Time (Minutes since Midnight)')
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()  # Close the plot to free memory
        
        return {
            'analysis_result': analysis_result,
            'visualization': f'data:image/png;base64,{img_base64}',
            'predictions': prediction_results
        }
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-delivery-time/")
async def predict_delivery_time(data: DeliveryData):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])
        input_data['preferred_time'] = pd.to_datetime(input_data['preferred_time'], format='%H:%M').dt.time
        input_data['preferred_time_minutes'] = input_data['preferred_time'].apply(lambda x: x.hour * 60 + x.minute)

        # Make prediction
        if isinstance(model, np.ndarray):
            # If the model is a NumPy array, use it as a feature matrix and make the prediction manually
            prediction = model @ input_data['preferred_time_minutes'].to_numpy().T
        else:
            # If the model is a scikit-learn model, use the predict method
            prediction = model.predict(input_data[['preferred_time_minutes']])

        # Convert prediction to time format
        predicted_time = int(prediction[0])
        predicted_hours = predicted_time // 60
        predicted_minutes = predicted_time % 60

        return {
            'predicted_delivery_time': f'{predicted_hours}:{predicted_minutes:02d}'
        }
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))