import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE
LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
BATCH_SIZE = 8
IMAGE_SIZE = [320, 320]

@st.cache(allow_output_mutation=True)
def get_model():
    model = tf.keras.models.load_model("model.h5")
    return model

st.write("# Blindness Detection")
st.write("This web app detects diabetic retinopathy in fundus photographs")

def predict(img):

    def decode_img(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.image.resize(image, IMAGE_SIZE)

    def preprocess(image):
        image = tf.io.read_file(image)
        image = decode_img(image)
        return image

    image = list(tf.data.Dataset.list_files(img))

    ds = tf.data.Dataset.from_tensor_slices(image)

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)

    model = get_model()

    prediction = model.predict(ds)

    number = str(round(100 * prediction[0][np.argmax(prediction[0])], 2))
    predicted_label = f"This image is {LABELS[np.argmax(prediction)]} with confidence: {number}%"

    st.success(predicted_label)


f = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])

if f is not None:
    f1 = open('/usr/src/app/test.png', 'wb')
    f1.write(f.read())
    f1.close()
    image = Image.open('/usr/src/app/test.png')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    start = time.time()
    predict('/usr/src/app/test.png')
    st.info("Prediction time is {:.2f} seconds".format(time.time() - start))
