from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from typing import List
import json

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

# Define the data model
class DeliveryData(BaseModel):
    recipient_id: str
    preferred_time: str  # Assuming HH:MM format
    delivery_date: str
    delivery_time: str   # Assuming HH:MM format
    location: dict
    day_of_week: str
    special_event: str

process_data = APIRouter()

@process_data.post("/process-data/")
async def process_data_endpoint(data: List[DeliveryData]):
    try:
        # Load the JSON data
        with open('delivery.json', 'r') as file:
            delivery_data = json.load(file)

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

        # Group the data by location and find the optimal delivery time for each location
        df['location_key'] = df.apply(lambda row: f"{row['location']['city']}, {row['location']['region']} - {row['location']['postal_code']}", axis=1)
        location_groups = df.groupby('location_key')
        optimal_delivery_times = {}

        for location_key, group_df in location_groups:
            preferred_times = group_df['preferred_time_minutes'].tolist()
            optimal_delivery_time = min(preferred_times)
            optimal_delivery_times[location_key] = optimal_delivery_time

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
            'optimal_delivery_times': optimal_delivery_times,
            'visualization': f'data:image/png;base64,{img_base64}'
        }
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))