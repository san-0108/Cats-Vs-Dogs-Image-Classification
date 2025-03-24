import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('image-classification-model')  # Update this if using .h5

# Define class labels (modify as per your model)
class_labels = ['Cat', 'Dog', 'Other']

st.title('Image Classification - Cats vs Dogs')
st.write("Upload an image to classify")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((128, 128))  # Adjust based on model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Prediction
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100  # Convert to %

    # Display result
    st.write(f"**Prediction:** {predicted_class} ({confidence:.2f}%)")
