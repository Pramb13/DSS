import streamlit as st
import torch
import cv2
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from datetime import datetime
import os

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
THRESHOLD = 0.6  # Adjust based on testing
USER_CREDENTIALS = {"user": "123"}
ADMIN_CREDENTIALS = {"admin": "admin123"}

# Store session predictions
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

# Ensure misclassified images are stored for debugging
MISCLASSIFIED_DIR = "misclassified_images"
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

# Authentication Function
def authenticate(username, password, role):
    if role == "User" and username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True
    elif role == "Admin" and username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        return True
    return False

@st.cache_resource
def load_model():
    """ Load the image classification model and feature extractor """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, "cpu"

def preprocess_frame(frame, feature_extractor, device):
    """ Convert and preprocess a single frame for model input """
    image = Image.fromarray(frame)  # Convert NumPy array to PIL Image
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input size
    inputs = feature_extractor(images=image, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}  # Move to GPU if available

def get_prediction(model, inputs):
    """ Get model prediction and confidence score """
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)  # Normalize scores
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            prediction_score = probabilities[0, predicted_class_idx].item()

        return predicted_class_idx, prediction_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def run_live_detection():
    """ Runs real-time drowsiness detection using OpenCV """
    st.write("Starting live detection...")

    model, feature_extractor, device = load_model()
    if model is None or feature_extractor is None:
        st.error("Failed to load model.")
        return

    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        st.error("Error: Cannot access webcam.")
        return

    stop_signal = st.button("Stop Live Detection")  # Button to stop video feed

    frame_window = st.image([])  # Create an image container for live streaming

    while cap.isOpened() and not stop_signal:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        inputs = preprocess_frame(frame_rgb, feature_extractor, device)
        predicted_class_idx, prediction_score = get_prediction(model, inputs)

        # Display the prediction on the frame
        label = LABELS[predicted_class_idx] if predicted_class_idx is not None else "Unknown"
        confidence_text = f"Confidence: {prediction_score:.2f}" if prediction_score else "N/A"
        
        # Draw label on frame
        cv2.putText(frame_rgb, f"{label} ({confidence_text})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display live frame in Streamlit
        frame_window.image(frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

def sidebar():
    """ Sidebar authentication for users and admins """
    st.sidebar.title("Drowsiness Detection System")
    role = st.sidebar.radio("Select Role", ("User", "Admin"))
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if authenticate(username, password, role):
            st.session_state["authenticated"] = True
            st.session_state["role"] = role
            st.sidebar.success(f"Logged in as {role}")
        else:
            st.sidebar.error("Invalid credentials. Please try again.")

def main():
    """ Main app logic """
    st.title("Real-Time Drowsiness Detection")
    st.markdown("This application detects drowsiness using a deep learning model.")
    sidebar()
    
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        st.warning("Please log in to continue.")
        return
    
    role = st.session_state.get("role", "User")
    
    if role == "User":
        if st.button("Start Live Detection"):
            run_live_detection()

    else:  # Admin Panel
        st.title("Admin Dashboard")
        st.write("Below are the recorded predictions with date and time:")
        
        if st.session_state["predictions"]:
            df = pd.DataFrame(st.session_state["predictions"])
            st.dataframe(df)
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
