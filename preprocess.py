from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.tolist()
