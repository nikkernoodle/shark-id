#User -> input picture
#Picture→ preprocessed
#	→ run through model

import streamlit as st
import tensorflow as tf
from PIL import Image
from shark_id.predict import predict_image
from io import BytesIO
import numpy as np
import requests


## Streamlit app
#st.title("Shark-ID")
#st.text("Upload an image to predict the shark species.")

#now lokal, later we will put it to google cloud
url = 'http://0.0.0.0:8501'

# Shark-ID front

buffer_image = st.file_uploader('Upload an Image')
if buffer_image is not None:
    image = Image.open(buffer_image)
    image_array= np.array(image) # if you want to pass it to OpenCV
    st.image(image_array, caption="The caption", use_column_width=True)



#prediction = predict_image(image)


# model
#load_model takes model_path as an argument
#we need to be able to give the model path as an argument
#to predict the image we need to then create something else using predict_image


# Make the prediction
if st.button('Predict'):
    #prediction = predict_image(image)
    st.text("This is the prediction:")
    #st.write(prediction)
    img_bytes = buffer_image.getvalue()
    res = requests.post(url + "/predict_file", files={'file': img_bytes})
    print(res)
    if res.status_code == 200:
        ### Display the image returned by the API
        st.text(res.json())
    else:
        st.markdown("**Oops**, something went wrong 😓 Please try again.")
        print(res.status_code, res.content)


# get prediction from model

### Create a native Streamlit file upload input




#
## Upload and display the image
#image = st.file_uploader('Upload an Image')
#if image is not None:
#    image = Image.open(image)
#    st.image(image, caption='Uploaded Image', use_column_width=True)
#



# Streamlit-App
#def main():
#    st.title("Shark UI Skeleton")
#    uploaded_file = st.file_uploader("choose image", type=["jpg", "jpeg", "png"])
#    if uploaded_file is not None:
#        image_path = "path/to/save/uploaded/image.jpg"
#        with open(image_path, "wb") as f:
#            f.write(uploaded_file.read())
#        st.image(image_path, caption="uploaded image", use_column_width=True)
#        preds = predict(image_path)
#        st.subheader("Prediction:")
#        for pred in preds:
#            st.write(f"{pred[1]}: {pred[2]*100:.2f}%")

#if __name__ == '__main__':
#    main()
