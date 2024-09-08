# In your main.py file
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import process_data

app = FastAPI()

# Allowing CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your front-end domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # This allows all headers
)

app.include_router(process_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
