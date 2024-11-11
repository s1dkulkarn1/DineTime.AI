import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
import os

# Set OpenAI API key
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

    /* Custom styling for subheader bubble */
    .custom-subheader {
        color: white;
        background-color: #E69561;  /* Muted orange background */
        padding: 10px 20px;
        border-radius: 15px;
        font-size: 1.25em;
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
        display: inline-block;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Text, label, and placeholder styling */
    .stTextInput label, .stButton button, .stMarkdown, .css-16huue1 {
        color: #333333 !important;  /* Black text for readability */
        font-family: 'Helvetica', sans-serif;
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

st.title("🍽️DineTime.Ai🍽️")
st.markdown("Plan your restaurant visit with real-time data and content summaries.")

# Custom subheader with orange bubble
st.markdown('<div class="custom-subheader">Enter Your Restaurant Details</div>', unsafe_allow_html=True)

# Section: Input Restaurant Details
restaurant_name = st.text_input("Restaurant Name", placeholder="e.g., Panda Express, Chipotle, Taco Bell, Subway")
location = st.text_input("Location", placeholder="e.g., Sunnyvale, Santa Clara, Palo Alto, Fremont")
visit_time = st.text_input("Planned Visit Time", placeholder="Enter your desired time (e.g. 12:00 PM)")

# URL input for restaurant website
url = st.text_input("Restaurant Website URL", placeholder="https://example.com (optional)")

# Function to fetch and parse content from the provided URL
# Code generated with assistance from ChatGPT, OpenAI, 2024.
def get_web_content(url):
    time_related_content = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract time-related content
        paragraphs = soup.find_all(['p', 'div', 'span'])
        for p in paragraphs:
            text = p.get_text().strip()
            # Filter for keywords like "hours," "open," "close," etc.
            if any(keyword in text.lower() for keyword in ["hours", "open", "close", "time", "am", "pm"]):
                time_related_content += text + '\n'

        return time_related_content if time_related_content else "No time-related information found."
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the content: {e}")
        return None

# Function to generate a recommendation based on fetched hours and visit time
# Code generated with assistance from ChatGPT, OpenAI, 2024.
def generate_recommendation(time_related_content, visit_time):
    prompt = f"""Based on the restaurant's operating hours:\n\n{time_related_content}\n\n
    and the user's planned visit time: {visit_time}, provide a recommendation on whether this is a suitable time to visit the restaurant."""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing visit recommendations based on restaurant hours."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Single button to fetch content, summarize, and provide recommendation
# Code generated with assistance from ChatGPT, OpenAI, 2024.
if st.button("Provide Recommendation"):
    if url:  # Only fetch content if a URL is provided
        display_content = get_web_content(url)
        if not display_content:
            display_content = "No time-related information found."
    else:
        display_content = "No URL provided, so no specific hours data is available."

    # Generate recommendation based on visit time and restaurant hours (if available)
    recommendation = generate_recommendation(display_content, visit_time)
    st.write("**Visit Recommendation:**")
    st.write(recommendation)
