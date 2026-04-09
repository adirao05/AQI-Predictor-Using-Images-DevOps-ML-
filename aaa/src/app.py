import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model_cnn import CNNModel as CNN

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'aaa', 'outputs', 'models', 'cnn_aqi.pth')
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_aqi_label(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

st.title("AQI Prediction from Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        prediction = output.item()
    label = get_aqi_label(prediction)
    st.subheader(f"Predicted AQI: {round(prediction, 2)}")
    st.subheader(f"Category: {label}")