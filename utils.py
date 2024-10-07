# Importing Necessary Libraries
import tensorflow as tf
import numpy as np
from PIL import Image

# Cleaning the image
def clean_image(image):
    image = np.array(image)
    
    # Resizing the image using the updated resampling method
    image = Image.fromarray(image).resize((512, 512), Image.Resampling.LANCZOS)
    image = np.array(image)
        
    # Adding batch dimensions to the image
    # Ensure that we are always using 3 channels (RGB)
    if image.shape[-1] > 3:
        image = image[:, :, :3]
    
    image = image[np.newaxis, :, :, :]  # Adding batch dimension
    
    return image

# Getting the prediction from the model
def get_prediction(model, image):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Inputting the image to keras generators
    test = datagen.flow(image)
    
    # Predict from the image
    predictions = model.predict(test)
    predictions_arr = np.argmax(predictions, axis=1)  # Ensure predictions_arr is the index
    
    return predictions, predictions_arr

# Making the final results
def make_results(predictions, predictions_arr):
    result = {}
    class_names = ["Healthy", "Multiple Diseases", "Rust", "Scab"]
    
    if predictions_arr in range(len(class_names)):
        status = class_names[int(predictions_arr)]
        prediction_percentage = int(predictions[0][int(predictions_arr)].round(2) * 100)
        result = {"status": f"has {status}",
                  "prediction": f"{prediction_percentage}%"}
    else:
        result = {"status": "Unknown",
                  "prediction": "0%"}
    
    return result
