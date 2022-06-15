# References: https://bit.ly/3zAxltV, https://bit.ly/3tAu2z3
import streamlit as st
import pandas as pd
import numpy as np
from  PIL import Image
import boto3
import json
import requests
import os
from botocore.exceptions import ClientError

def main():
    s3 = boto3.client('s3')
    bucket = 'sagemaker-us-east-1-328945632120'

    key = "streamlit-images"
    key_name = f"{key}/dogImage.jpg"


    api_url = "https://q5giae49dd.execute-api.us-east-1.amazonaws.com/dev/predictdogimages"
    headers = {'Content-type': 'application/json'}


    #Add a header and expander in side bar
    st.write("""
    # Welcome to Habitat for Canine's (H4C) Automated Triage App

    Use this web app to submit images of lost and found dogs in your neighborhood.
    Also tell us the street address where you last found the dog. We will take if from there 
    and work to get this cute dog 🐶 into the appropriate shelter.
    From there, it can eventually find a **forever home**!


    """)
    st.sidebar.markdown('<p class="font">Lost & Found Dog Triage App</p>', unsafe_allow_html=True)
    # with st.sidebar.expander("About the App"):
    #      st.write("""
    #         Blah...
    #      """)


    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)


    # Add file uploader to allow users to upload photos
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])


    with st.form("dog_sighting_form", clear_on_submit=False): 
        if uploaded_file is not None:
            # print(uploaded_file)
            image = Image.open(uploaded_file)
            st.markdown('<p style="text-align: left;">Found Dog</p>',unsafe_allow_html=True)
            st.image(image,width=300)
            # To See details
            file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type,
                            "filesize":uploaded_file.size}
            st.write(file_details)


            #Saving upload
            with open("dogImage.jpg","wb") as f:
                f.write((uploaded_file).getbuffer())
            try:
                s3.upload_file("dogImage.jpg", bucket, key_name)
                # success_write = "Successfully uploaded now predicting..."
                url = f"https://{bucket}.s3.amazonaws.com/{key_name}"
                payload = json.dumps({
                    "url": url
                })
                r = requests.post(url=api_url, data=payload, headers=headers)
                response = json.loads(r.json()['body'])
                message = f"This dog might either be a **{response['first_predicted_dog']}** or a **{response['second_predicted_dog']}**"
                st.write(message)
                st.write("**Thanks for being a great citizen 🐕!!** Now, kindly share some information about its location and we will take it from here.")
                # st.write(response['second_confidence'], response['second_predicted_dog'])

                # References: https://bit.ly/3MW608z, https://bit.ly/3OfV9HE
                name = st.text_input("Enter your name") 
                email = st.text_input("Enter your email so we can contact you") 
                address = st.text_input("Enter the last address you saw the dog") 
                zip_code = st.text_input("A Zip code would be useful as well") 
                message = st.text_area("Any additional details?") 
                button_check = st.form_submit_button("Submit")
                if button_check:
                    st.success('Awesome, you successfully submitted the details!')


            except ClientError as e:
                logging.error(e)
                st.write("Sorry we encountered an error with this file")


if __name__ == '__main__':
	main()
