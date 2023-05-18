import time
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

# Link emoji https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Nhận diện bệnh cây cà chua", page_icon= ":tomato:",layout = 'wide')

# Navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title = "",
        options = ["Home", "Guide", "About", "Contact"],
        icons = ["house", "book", "file-earmark-person", "envelope"],
        default_index = 0,
        styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "blue", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#DDE0E6"},
            }
    )

# Load model
model = tf.keras.models.load_model("ModelTomato.h5")

# Select mode 
#----------------------------------Home---------------------------------------
if selected == "Home":
    st.title('Nhận diện bệnh cây cà chua')
    uploaded_file = st.file_uploader("Chọn ảnh muốn nhận diện tại đây:", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
         # Convert file size
        imga = image.load_img(uploaded_file,target_size=(256,256))
        st.image(imga, channels="RGB")
        img = image.load_img(uploaded_file,target_size=(100,100))

        # Convert to array
        img = img_to_array(img)
        img = img.reshape(1,100,100,3)
        img = img.astype('float32')
        img = img/255

        # Button detection
        Button_detect = st.button("Detect")
        if Button_detect:
            with st.spinner("Please wait, Running!!!"):
                time.sleep(2)
            prediction = model.predict(img).argmax()
            y_pred = model.predict(img)
            # Phân loại bệnh
            if prediction == 1:
                st.write("**Cây cà chua của bạn bình thường**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 2:
                st.write("**Cây cà chua của bạn bị Cháy lá**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 3:
                st.write("**Cây cà chua của bạn bị Đốm do nhện đỏ gây ra**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 4:
                st.write("**Cây cà chua của bạn bị Đốm lá do vi khuẩn Septoria**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 5:
                st.write("**Cây cà chua của bạn bị Đốm vi khuẩn**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 6:
                st.write("**Cây cà chua của bạn bị Đốm vòng**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 7:
                st.write("**Cây cà chua của bạn bị Héo muộn**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 8:
                st.write("**Cây cà chua của bạn bị Khảm lá do vi khuẩn Mosaic**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 9:
                st.write("**Cây cà chua của bạn bị Mốc lá**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
            elif prediction == 10:
                st.write("**Cây cà chua của bạn bị Vàng xoăn lá do vi khuẩn Curl**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

    
#----------------------------------Guide---------------------------------------
if selected == "Guide":
    st.title('You choice Guide')
#----------------------------------About---------------------------------------
if selected == "About":
    st.title('You choice About')
#----------------------------------Contact---------------------------------------
if selected == "Contact":
    st.title('You choice Contact')
st.title('LuuQuangHoi')