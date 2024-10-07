# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
import os
from tensorflow.keras.applications import Xception, DenseNet121

# Define the Model Architecture
def create_model():
    # Xception Model
    xception_model = tf.keras.Sequential([
        Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.Sequential([
        DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))

    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Loading the Model and saving to cache
@st.cache_resource
def load_model(path):
    model = create_model()
    try:
        # Loading the Weights of the Model
        model.load_weights(path)
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"File not found: {path}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    return model

# Function to Provide Cure Information and General Plant Remedies
def get_cure_info(disease):
    general_remedies = (
        "General Plant Care Tips:\n"
        "- Ensure your plant gets adequate sunlight.\n"
        "- Water your plant appropriately, avoiding both under and over-watering.\n"
        "- Use nutrient-rich soil and consider adding compost or fertilizers periodically.\n"
        "- Regularly check your plant for pests or diseases and take early action if needed."
    )
    
    cure_dict = {
        'Healthy': 'No action needed. Your plant is healthy!',
        'Disease A': 'Cure for Disease A: Apply fungicide and ensure proper watering.',
        'Disease B': 'Cure for Disease B: Remove affected leaves and treat with insecticide.',
        'Disease C': 'Cure for Disease C: Improve soil drainage and apply organic compost.',
        'Disease D': 'Cure for Disease D: Use disease-resistant plant varieties and apply appropriate fertilizer.'
    }
    
    specific_cure = cure_dict.get(disease, "No specific cure information available.")
    return f"{specific_cure}\n\n{general_remedies}"

# Removing Menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Verify and Print Current Working Directory
st.write("Current Directory:", os.getcwd())

# Loading the Model
model = load_model('model.h5')

# Check if the model was loaded successfully
if model is None:
    st.stop()  # Stop further execution if model loading failed

# Title and Description
st.title('Plant Disease Detection')
st.write("Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not")

# Setting the files that can be uploaded
uploaded_file = st.file_uploader("Choose an Image file", type=["png", "jpg"])

# If there is an uploaded file, start making predictions
if uploaded_file is not None:
    # Display progress and text
    progress = st.text("Crunching Image")
    my_bar = st.progress(0)
    
    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    
    # Correcting the resize method
    st.image(np.array(Image.fromarray(np.array(image)).resize((700, 400), Image.Resampling.LANCZOS)), width=None)
    my_bar.progress(40)
    
    # Cleaning the image
    image = clean_image(image)
    
    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(70)
    
    # Making the results
    result = make_results(predictions, predictions_arr)
    
    # Removing progress bar and text after prediction is done
    progress.empty()
    my_bar.empty()
    
    # Show the results
    disease = result['status']
    st.write(f"The plant is {disease} with {result['prediction']} prediction.")
    
    # Provide cure information
    cure_info = get_cure_info(disease)
    st.write(f"Cure Information: {cure_info}")
