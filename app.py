import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2

# Load and preprocess the image
def model_prediction(image_path):
    model = tf.keras.models.load_model("cnn_model.keras")
    img=cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H,W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype('float32')
    img = img/255.0
    img = img.reshape(1, H, W, C)
    
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

st.sidebar.title('Plant Disease Detection System for Sustainable Agriculture')
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition'])

from PIL import Image
img = Image.open('Disease.png')
st.image(img)

if(app_mode == 'Home'):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    
elif(app_mode == "Disease Recognition"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded image to the local directory
        save_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

    # Open image using PIL
    test_image = Image.open(uploaded_file)

    # Button to display the image
    if st.button("Show Image"):
        # Resize image to ~5cm x 5cm (189x189 pixels)
        resized_image = test_image.resize((189, 189))

        # Display the resized image
        st.image(resized_image, use_column_width=False, width=200)
        st.caption("Uploaded Leaf Image for Disease Detection")
    
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(save_path)
        print(result_index)
    
        class_name = [
'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        st.success("Model is predicting that it is a {}".format(class_name[result_index]))