### Calorie calculator app
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()  ### loads all the environment variables

import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load google gemini pro vision api and get response

def get_gemini_response(input,image,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,image[0],prompt])
    return response.text

#below function code from google bard to upload the image & read the data
def input_image_setup(uploaded_file):
    #Check if the file has been uploaded
    if uploaded_file is not None:
        #Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts=[
            {
                "mime_type": uploaded_file.type, #get the mime type of the uploaded file
                "data": bytes_data  #gemini will be expecting this data of the image
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

## initialize our streamlit app
st.set_page_config(page_title="Gemini calorie calculator app")
st.header("Gemini calorie calculator app")
input = st.text_input("Input Prompt:",key="input")
uploaded_file = st.file_uploader("Choose an image...",type = ["jpg","jpeg","png"])
image = ""
if uploaded_file is not None:
    image= Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image.",use_column_width=True)

submit = st.button("Get total calories")

input_prompt = """
You are a nutritionist where you need to see the food items from the image and calculate the total calories.
            If the image is of non-veg platter, then all items will be non-veg only.
            If the image is of veg platter,then all items will be vegeterian only.
            Also provide the details of every food items with calorie intake in below format
            
            1. Item 1 - no of calories
            2. Item 2 - no of calories
            ------
            ------

"""

# if submit button is clicked
if submit:
    image_data=input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt,image_data,input)
    st.subheader("The response is:")
    st.write(response)