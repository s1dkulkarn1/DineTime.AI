import os
import openai
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path

# Set your OpenAI API key (ensure this is secure in production environments)
openai.api_key = os.environ["OPENAI_API_KEY"]

# uiux changes
st.markdown("""
    <style>
    /* Main background and font */
    .main, .stApp {
        background-color: #F2E0C9;  /* Soft beige background for entire app */
        font-family: 'Helvetica', sans-serif;
    }

    /* Title styling */
    h1 {
        color: #222222;  /* Dark black for title */
        font-family: 'Helvetica', sans-serif;
    }

    /* Custom styling for subheader */
    .custom-subheader {
        color: white;
        background-color: #E69561;  /* Muted orange background */
        padding: 8px 15px;
        border-radius: 10px;
        font-size: 1.25em;
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
        display: inline-block;
    }

    /* Text, label, and placeholder styling */
    .stTextInput label, .stFileUploader label, .stButton button, .stMarkdown, .css-16huue1 {
        color: #333333 !important;  /* Black text for readability */
        font-family: 'Helvetica', sans-serif;
    }

    /* Enhanced Caption styling for images with black bubble */
    .stImage img + div p { 
        color: #FFFFFF !important;  /* White text for contrast */
        font-size: 1.1em;            /* Slightly larger font for better readability */
        font-weight: bold;           /* Bold font for clarity */
        background-color: #333333;   /* Black background for the bubble */
        padding: 6px 12px;           /* Padding around text for visibility */
        border-radius: 8px;          /* Rounded corners for aesthetic */
        text-align: center;          /* Centered caption text */
        display: inline-block;       /* Ensures bubble wraps only the text */
        margin-top: 5px;             /* Small margin above the caption */
    }

    /* Button customization */
    .stButton button {
        background-color: #E69561;  /* Muted orange button */
        color: white;
        font-weight: bold;
        padding: 8px 20px;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #D17B4A;  /* Darker orange on hover */
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #222222 !important;  /* Dark sidebar */
        color: white !important;  /* White text for contrast on black sidebar */
    }

    </style>
""", unsafe_allow_html=True)

# Define the file path to the historical data CSV using an environment variable or fallback
file_path = Path("118i-tutorial/pages/data/Data.csv")

# Load the historical data and calculate average wait times by hour
try:
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data['Hour'] = data['Datetime'].dt.hour
    average_wait_by_hour = data.groupby('Hour')['Estimated Wait Time (minutes)'].mean()
except FileNotFoundError:
    st.error(f"Historical data file not found at {file_path}. Please ensure the CSV file path is correct.")
    st.stop()

# Streamlit app title and description
st.title("🍽️Restaurant Wait Time Estimator (IN-PERSON)🍽️")
st.markdown("Combining historical data and real-time line analysis for more accurate wait time estimates.")

# Custom subheader with orange background
st.markdown('<div class="custom-subheader">Upload an Image of the Restaurant Line</div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Upload an image (JPEG, PNG) of the line at the restaurant", type=["jpg", "jpeg", "png"])

# Function to blur faces in an image
def blur_faces(image: Image.Image) -> Image.Image:
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(opencv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_region = opencv_image[y:y+h, x:x+w]
        face_region = cv2.GaussianBlur(face_region, (51, 51), 30)
        opencv_image[y:y+h, x:x+w] = face_region

    return Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

# Function to generate feedback based on image description and historical data
def generate_feedback_with_data(image_description, current_hour, average_wait_by_hour):
    avg_wait_time = average_wait_by_hour.get(current_hour, "No data available")
    prompt = (
        f"Based on historical data, the average wait time around {current_hour}:00 is approximately "
        f"{avg_wait_time} minutes. Given the following description of the current line: {image_description}, "
        "please provide an adjusted wait time estimate and any additional operational insights."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are an assistant providing wait time estimations based on data and real-time conditions."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Display image and provide feedback if an image is uploaded
if uploaded_image:
    image = Image.open(uploaded_image)
    blurred_image = blur_faces(image)
    st.image(blurred_image, caption="Uploaded Image with Blurred Faces", use_column_width=True)
    
    current_hour = datetime.now().hour
    image_description = "An image showing a long line of people waiting outside a popular restaurant."
    feedback = generate_feedback_with_data(image_description, current_hour, average_wait_by_hour)
    
    st.write("**Feedback on Wait Time (Enhanced with Historical Data):**")
    st.write(feedback)
