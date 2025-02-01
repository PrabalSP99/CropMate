from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import requests

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Request Model
class CropRequest(BaseModel):
    id: int
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Temperature: float
    Humidity: float
    pH: float
    Rainfall: float

@app.get("/")
def home():
    return {"message": "Hello from FastAPI backend"}

@app.post("/predict")
def predict(data: CropRequest):
    try:
        # Prepare input features
        features = np.array([[data.Nitrogen, data.Phosphorus, data.Potassium,
                              data.Temperature, data.Humidity, data.pH, data.Rainfall]])

        # Model prediction
        prediction = model.predict_proba(features)

        # Get top 5 predicted crops
        top5_classes = np.argsort(prediction[0])[-5:]
        top5_crops = [crop_dict[idx + 1] for idx in reversed(top5_classes)]

        # Prepare response
        result = {
            "message": "Prediction successful",
            "id": data.id,
            "Crop1": top5_crops[0],
            "Crop2": top5_crops[1],
            "Crop3": top5_crops[2],
            "Crop4": top5_crops[3],
            "Crop5": top5_crops[4],
        }

        # Send data to external Crop API
        crop_api_url = "http://localhost:4999/crop"
        crop_data = result.copy()  # Send the same data
        response = requests.post(crop_api_url, json=crop_data)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to save crop data: {response.text}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
