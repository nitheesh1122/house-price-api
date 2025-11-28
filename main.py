from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr

# Load ML model
model = joblib.load("house_model.pkl")

# FastAPI app
app = FastAPI()

class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.get("/")
def home():
    return {
        "message": "Welcome to the House Price Prediction API!",
        "gradio_ui": "/gradio",
        "api_docs": "/docs"
    }

@app.post("/predict")
def predict(input: Input = Input()):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}

# --------- Gradio UI ---------
def predict_price(med_inc, house_age, rooms, bedrooms, population, households, lat, lon):
    data = [med_inc, house_age, rooms, bedrooms, population, households, lat, lon]
    pred = model.predict([data])[0]
    return f"Predicted House Price: ${pred:,.2f}"

gradio_app = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Median Income"),
        gr.Number(label="House Age"),
        gr.Number(label="Rooms"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Population"),
        gr.Number(label="Households"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🏡 House Price Predictor",
    description="Enter house details to get a price prediction"
)

app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
