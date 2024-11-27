import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import time
import os
import gdown

def download_model_from_drive():
    url = 'https://drive.google.com/file/d/1W54xonnkYjeKqqnltCe_iz2tFX7OjCGb/view?usp=sharing'  # Replace with your file's ID
    output = 'best_model.pth'
    gdown.download(url, output, quiet=False)
    return output

# Load the pre-trained model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2) 
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define image transformations for prediction
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Function to predict the class of the image
def predict_image(image, model, class_names):
    image_tensor = data_transforms(image).unsqueeze(0)  # Convert image to tensor
    outputs = model(image_tensor)
    _, preds = torch.max(outputs, 1)  # Get predicted class
    return class_names[preds.item()]

# Function to capture a frame from the webcam
def capture_frame_from_camera(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# Function to crop the image to a fixed region (x, y, width, height)
def crop_image(image, x, y, width, height):
    np_image = np.array(image)
    cropped_image = np_image[y:y+height, x:x+width]  # Crop the image
    return Image.fromarray(cropped_image)

def main():
    st.title("AI Detection for Clamp")

    model_path = download_model_from_drive 
    if not os.path.exists(model_path):
        st.error("Error in the 'best_model.pth' file's path.\nEnsure the path of the file is in the same directory")
        return

    model = load_model(model_path)
    class_names = ["Not Okay", "Okay"]
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Could not open the camera. Please check your camera connection.")
        return

    image_placeholder = st.empty()
    result_placeholder = st.empty()

    try:
        while True:
            # Capture frame from the camera
            image = capture_frame_from_camera(cap)
            if image is None:
                st.warning("No frame captured. Retrying...")
                time.sleep(5)
                continue

            # Crop the image to the fixed region (adjust x, y, width, height as needed)
            cropped_image = crop_image(image, 200, 100, 200, 200)  # Cropping the image

            # Predict the class of the cropped image
            with st.spinner("Classifying..."):
                prediction = predict_image(cropped_image, model, class_names)

            # Update background color based on prediction
            if prediction == "Not Okay":
                st.markdown(
                    """
                    <style>
                    .stApp {
                        background-color: red;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
            elif prediction == "Okay":
                st.markdown(
                    """
                    <style>
                    .stApp {
                        background-color: green;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

            # Display the cropped image without caption
            image_placeholder.image(cropped_image, use_container_width=True)

            # Display the prediction result
            result_placeholder.success(f"Prediction: {prediction}")

            # Sleep for a while before capturing the next frame
            time.sleep(5)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()