import gradio as gr
from gradio import components as gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

print("Starting script...")

# Set the GRADIO_SERVER_PORT environment variable
os.environ["GRADIO_SERVER_PORT"] = "7850"

# Explicitly define the input shape for MobileNetV2
input_shape = (224, 224, 3)

# Load the pre-trained MobileNetV2 model with the specified input shape
base_model = MobileNetV2(weights='imagenet', include_top=True, input_shape=input_shape)

shot_types = ["upper_body", "cowboy", "close_up", "portrait", "above", "full_body"]

def detect_damaged_eyes(image_path, blur_threshold):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    value = 0  # Initialize the value variable here

    for (ex, ey, ew, eh) in eyes:
        roi_gray = gray[ey:ey+eh, ex:ex+ew]
        value = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        if value < blur_threshold:  # Use the dynamic threshold
            return True
    return False


def classify_shot_type(features):
    # Normalize the features to [0, 1] range
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features))
    
    # Check the concentration of features
    top_quarter = np.mean(normalized_features[:, :56])
    upper_half = np.mean(normalized_features[:, :112])
    lower_half = np.mean(normalized_features[:, 112:])
    center = np.mean(normalized_features[112:160, 64:160])
    
    if top_quarter > 0.6:
        return "portrait"
    elif center > 0.6:
        return "close_up"
    elif upper_half > lower_half + 0.1 and top_quarter < 0.6:
        return "upper_body"
    elif lower_half > upper_half + 0.1:
        return "above"
    elif upper_half > 0.5 and lower_half < 0.3:
        return "cowboy_shot"
    else:
        return "full_body"  # default to full body

def resize_and_crop(img, target_size=(224, 224)):
    # Calculate aspect ratio
    aspect = img.width / img.height

    # Resize while maintaining aspect ratio
    if aspect > 1:
        # Landscape orientation - wide image
        width = int(aspect * target_size[1])
        img = img.resize((width, target_size[1]))
    else:
        # Portrait orientation - tall image
        height = int(target_size[0] / aspect)
        img = img.resize((target_size[0], height))

    # Crop to the desired size
    left_margin = (img.width - target_size[0]) / 2
    bottom_margin = (img.height - target_size[1]) / 2
    right_margin = left_margin + target_size[0]
    top_margin = bottom_margin + target_size[1]

    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    return img

def classify_image(img_path):
    # Load the image
    original_img = image.load_img(img_path)
    
    # Create a copy of the original image for resizing and cropping
    img = original_img.copy()
    
    # Resize and crop the image for classification
    img = resize_and_crop(img, target_size=(224, 224))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Extract features using MobileNetV2
    features = base_model.predict(x)
    
    # Classify the shot type based on the extracted features
    shot_type = classify_shot_type(features)
    
    return shot_type

def segregate_images(source_folder, upper_body, cowboy, close_up, portrait, above, full_body, blur_threshold):
    # Create necessary directories within the source folder
    folders = ["damaged_eyes"] + shot_types
    for folder in folders:
        if not os.path.exists(os.path.join(source_folder, folder)):
            os.mkdir(os.path.join(source_folder, folder))

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                if detect_damaged_eyes(img_path, blur_threshold):  # Pass the dynamic threshold
                    os.rename(img_path, os.path.join(source_folder, "damaged_eyes", file))
                else:
                    classification = classify_image(img_path)
                    os.rename(img_path, os.path.join(source_folder, classification, file))
    
    return f"Images from {source_folder} have been processed."

def gradio_ui():
    print("Initializing Gradio UI...")
    interface = gr.Interface(
        fn=segregate_images,
        inputs=[
            gc.Textbox(label="Angle Segregator: Provide the path to your source folder here. All images to be segregated need to be present in the source folder. Subfolders will not be scanned."),
            gc.Checkbox(label="Upper Body"),
            gc.Checkbox(label="Cowboy"),
            gc.Checkbox(label="Close-Up"),
            gc.Checkbox(label="Portrait"),
            gc.Checkbox(label="Above"),
            gc.Checkbox(label="Full Body"),
            gc.Slider(minimum=0, maximum=100, default=50, label="Blur Threshold for Eyes")  # Add the slider
        ],
        outputs=gc.Textbox(),
        live=False
    )
    print("Launching Gradio...")
    interface.launch()

print("Executing main...")
if __name__ == "__main__":
    gradio_ui()