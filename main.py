from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("house_model.pkl")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input = Input()):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}

# -----------------------------
# Gradio Prediction Function
# -----------------------------
def predict_price(med_inc, house_age, rooms, bedrooms, population, households, lat, lon):
    data = [med_inc, house_age, rooms, bedrooms, population, households, lat, lon]
    pred = model.predict([data])[0]
    return f"Predicted House Price: ${pred:,.2f}"

# -----------------------------
# Gradio UI
# -----------------------------
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
    title="🏡 California House Price Predictor",
    description="Enter house details to get a price prediction"
)

# Mount Gradio UI inside FastAPI
app.mount("/gradio", WSGIMiddleware(gradio_app))

# -----------------------------
# Start App Locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
