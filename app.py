import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


model = load_model("digit_recognition_model.keras")

st.title("Digit Recognition")

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

st.write("Draw the number and click the button below")

if st.button("Recognize digit"):
    if canvas_result.image_data is not None:

        st.image(canvas_result.image_data, caption="original image")

        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        img = img.resize((28, 28))
        st.image(img, caption="28x28 (image for model)")

        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predicted_value = model.predict([img_array])
        predicted_value = predicted_value.argmax(axis=1)
        st.write(f"Predicted value is {predicted_value}")

    else:
        st.error("Paint the number")
