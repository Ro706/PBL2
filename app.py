import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("fire_detection_model.h5")

st.set_page_config(page_title="Forest Fire Detection", layout="centered")

st.title("🌲🔥 Forest Fire Detection")

# Show Earth rotating image
earth_image_url = "https://static.vecteezy.com/system/resources/thumbnails/002/019/623/large/a-transparent-earth-rotates-free-video.jpg"
st.markdown(
    f"<center><img src='{earth_image_url}' alt='Rotating Earth' width='300'></center>",
    unsafe_allow_html=True,
)

st.write("Upload an image of the forest area to detect wildfire presence.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Input Image", use_column_width=True)

    img_for_model = img.resize((64, 64))
    img_array = np.array(img_for_model) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = "🔥 Wildfire Detected" if prediction > 0.5 else "✅ No Wildfire"

    st.markdown(
        f"<h3 style='text-align: center; color: {'red' if prediction > 0.5 else 'green'};'>{result}</h3>",
        unsafe_allow_html=True,
    )
