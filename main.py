import streamlit as st
import tensorflow as tf
import numpy as np


## Tensorflow Model Prediction
def model_prediction(test_image):
    
    model = tf.keras.models.load_model('trained_model.keras')
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    

    input_arr = np.expand_dims(input_arr, axis=0)
    

    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

## Sidebar
st.sidebar.title("Dashboad")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About', 'Disease Recognition'])
if (app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path=('image.jpg')
    st.image(image_path,use_column_width=[True])
    st.markdown("""
# Welcome to the Plant Detection System

Welcome to the Plant Detection System, your reliable tool for accurate plant identification and disease recognition. Our innovative platform leverages cutting-edge technology to help you identify plants and potential diseases through simple photo uploads.

## Our Mission

Our mission is to empower individuals and organizations to make informed decisions about plant health by providing accurate and easy-to-use plant recognition tools. We aim to promote sustainable agriculture and environmental stewardship through technology.

## How It Works

Our system is designed to be simple and user-friendly. Follow these easy steps to get started:
1. Take a clear photo of the plant leaf you want to identify.
2. Upload the image using the provided upload option.
3. Let our advanced machine learning model analyze the image and provide results.

**Note:** The system works best when clear, well-lit images of the plant leaf are provided.

## Why Choose Us?

- **Accuracy:** Our model is trained on a large database of plant species and diseases to ensure precise identification.
- **Speed:** Get your results within seconds of uploading an image.
- **Ease of Use:** The platform is designed with simplicity in mind, requiring no technical expertise from the user.
- **Expertise:** Developed by a team of professionals and students from the Technical University of Mombasa, with a passion for technology and environmental sustainability.

## About Us

Wee are a dedicated team from the **Technical University of Mombasa**, committed to using our knowledge of technology to improve plant health management. Our goal is to bridge the gap between agriculture and technology by providing accessible solutions for plant disease detection.

Visit our **About** page to learn more about our journey, vision, and the team behind the Plant Detection System.
""")
elif (app_mode == 'About'):
    st.header('About')
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset.

    #### Content
    1. **Train**: 70,295 images
    2. **Validation**: 17,572 images
    3. **Test**: 33 images
    """)   

   
elif app_mode == 'Disease Recognition':
    st.header('Disease Recognition')
    st.markdown("""
    Our plant disease recognition system is designed to help you quickly and accurately identify diseases in plant leaves.

    ### How It Works
    1. **Upload** a clear image of the plant leaf.
    2. Our model will **analyze** the image and compare it with known disease patterns.
    3. You will receive a **prediction** with the possible disease or plant health status.

    **Note:** Ensure the image is well-lit and focused on the leaf for the best results.

    ### Example Use Case
    If you are a farmer or gardener looking to detect potential diseases in your crops, simply take a photo of the affected leaf and upload it. Our system will guide you through the process and provide detailed information on the condition of the plant.
    """)
    test_image= st.file_uploader("Choose an Image:")
    if (st.button("Show Image")):
        st.image(test_image,use_column_width=[True])
    if (st.button("Predict")):
     with st.spinner("Please Wait...."):
        st.write('Our Prediction ')
        result_index= model_prediction(test_image)
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success("Model is Predicting a{}".format(class_name[result_index]))

