from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PIL import Image

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to get response from the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")
def get_gemini_response(input_text, image):
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text

# Streamlit setup
st.set_page_config(page_title="Gemini Image Demo")
st.header("Gemini Application")

input_text = st.text_input("Input Prompt:", key="input")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Tell me about the image")

if submit:
        response = get_gemini_response(input_text, image)
        st.subheader("The Response is")
        st.write(response)
