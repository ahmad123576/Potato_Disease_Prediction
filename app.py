import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = 'D:\\ML\\potato_project\\potato.h5'  # Path to your .h5 model file
model = load_model(MODEL_PATH)

# Class labels (must match the training order)
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

# App title
st.title("Potato Leaf Disease Classifier 🌿")
st.write("Upload a potato leaf image and get the predicted disease class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the image
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image_resized = image_pil.resize((224, 224))  # Match model input size
        img_array = np.array(image_resized) / 255.0   # Normalize
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Show prediction
        st.success(f"Prediction: **{predicted_class}**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
