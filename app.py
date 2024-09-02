import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the mapping from crops to their respective models
crop_to_model = {
    "Apple": "model1.h5",
    "Blueberry": "model5.h5",
    "Cherry": "model1.h5",
    "Corn": "model2.h5",
    "Grape": "model2.h5",
    "Orange": "model2.h5",
    "Peach": "model3.h5",
    "Pepper": "model3.h5",
    "Potato": "model3.h5",
    "Raspberry": "model4.h5",
    "Soybean": "model4.h5",
    "Squash": "model4.h5",
    "Strawberry": "model5.h5",
    "Tomato": "model5.h5",
    "Alstonia Scholaris": "model2.h5",
    "Sugarcane": "model4.h5",  # Ensure this is correctly mapped
    # Add any other crops and ensure they are mapped correctly
}

# Define the paths to the model files
model_paths = {
    "model1.h5": r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model.h5',
    "model2.h5": r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model2.h5',
    "model3.h5": r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model3.h5',
    "model4.h5": r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model4_v2.h5',
    "model5.h5": r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model5.h5',
}

# Load your Keras models
models = {}
for model_name, path in model_paths.items():
    try:
        model = tf.keras.models.load_model(path)
        models[model_name] = model
        print(f"Loaded model '{model_name}' from {path}")
    except Exception as e:
        print(f"Error loading model '{model_name}' from {path}: {e}")

# Labels for the predictions (ensure these match the model outputs)
crop_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
    "Alstonia Scholaris diseased", "Alstonia Scholaris healthy", "Arjun diseased", "Arjun healthy",
    "Citrus_Black spot", "Citrus_canker", "Citrus_greening", "Citrus_healthy", "Guava diseased",
    "Guava healthy", "Jamun diseased", "Jamun healthy", "Pomegranate diseased", "Pomegranate healthy",
    "Pongamia Pinnata diseased", "Pongamia Pinnata healthy", "Bael diseased", "Basil healthy",
    "Jatropa diseased", "Jatropa healthy", "Lemon diseased", "Lemon healthy", "Mango diseased",
    "Mango healthy", "Rose_Healthy_Leaf", "Rose_Rust", "Rose_sawfly_Rose_slug", "Soybean_healthy",
    "Sugarcane_Banded_Chlorosis", "Sugarcane_BrownRust", "Sugarcane_Brown_Spot", "Sugarcane_Grassy shoot",
    "Sugarcane_Pokkah Boeng", "Sugarcane_Sett Rot", "Sugarcane_Viral Disease", "Sugarcane_Yellow Leaf",
    "Tea_algal_spot", "Tea_brown_blight", "Tea_gray_blight", "Tea_healthy", "Tea_helopeltis", "Tea_red_spot",
    "Blueberry_healthy", "Cherry_healthy", "Cherry_Powdery_mildew", "Chinar diseased", "Chinar healthy",
    "Tulsi_bacterial", "Tusli_fungal", "Tulsi_healthy"
]

def getResult(image_path, model, labels):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Adjust target_size if needed
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = x.astype('float32') / 255.0  # Normalize the image data to 0-1

        predictions = model.predict(x)
        if len(predictions[0]) != len(labels):
            raise ValueError(f"Model output size ({len(predictions[0])}) does not match labels size ({len(labels)})")
        
        class_id = np.argmax(predictions)
        return labels[class_id]
    except Exception as e:
        print(f"Error in getResult function: {e}")
        return "Error in prediction", 500  # Return error response

@app.route('/', methods=['GET', 'POST'])
def index():
    unique_crops = sorted(crop_to_model.keys())
    if request.method == 'POST':
        try:
            file = request.files['file']
            crop = request.form.get('crop')
            
            if not crop:
                return "No crop selected", 400
            
            file_path = os.path.join(os.path.dirname(__file__), 'uploads', secure_filename(file.filename))
            file.save(file_path)
            
            model_name = crop_to_model.get(crop)
            if not model_name:
                return f"No model found for crop: {crop}", 400
            
            model = models.get(model_name)
            if not model:
                return f"Model {model_name} is not loaded properly.", 500
            
            print(f"Running prediction for {crop} using model {model_name}")
            result = getResult(file_path, model, crop_labels)
            print(f"Prediction result for {crop}: {result}")
            
            return result
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "An error occurred during prediction.", 500
    
    return render_template('index.html', crops=unique_crops)

if __name__ == '__main__':
    app.run(debug=True)