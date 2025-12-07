import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
@st.cache_resource
def load_cnn_model():
    model = load_model("bin_bags_classification_model.keras")
    return model

model = load_cnn_model()

# Streamlit UI
st.title("Bin Bags Classification")
st.write("Upload an image of a bin bags")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocessing (modify to match your model)
    img = img.resize((200, 200))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    st.success(f"Prediction: {predicted_class}")
