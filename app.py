import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('hwdr.h5')

st.header("Handwritten Digit Recognition Model")

uploaded_file = st.file_uploader("Upload an image of a handwritten digit", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image as a NumPy array
    image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        st.error("Error: Unable to read the uploaded image.")
    else:
        img = cv2.resize(img, (28, 28))
        img = np.invert(img)
        img = img.astype(np.float32) / 255.0

        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        output = model.predict(img)

        predicted_digit = np.argmax(output)

        st.subheader(f"Predicted number is : {predicted_digit}")
        st.image(uploaded_file, caption=f"Uploaded Image", width=300)

else:
    st.info("Please upload an image to predict the digit.")
