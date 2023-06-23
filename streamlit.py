import streamlit as st
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import av
import cv2
import os
import tempfile
model = YOLO('models/best.pt')
st.set_page_config(page_title="Computer vision", page_icon="🖥️")

def main():
    st.title("🪴")

    # Add file uploader
    file = st.file_uploader("Upload an image or video to diagnose your plant", type=["jpg", "jpeg", "png", "mp4"])

    if file is not None:
        if file.type.startswith('image'):
            # Process uploaded image
            img = Image.open(file)
            results= model(img)
            # results = model.predict(img, conf=0.25)

            # # Plot the results on the image
            res_plotted = results[0].plot(labels=True)
            st.image(res_plotted, caption='Predicted Image', use_column_width=True)


        elif file.type.startswith('video'):
            ##----version originale vvv
            video = cv2.VideoCapture(file)
            st.video(file)
            results= model(video)
            # Display predicted labels on the video frames
            annotated_video = results.render()  # Render annotated video frames with bounding boxes and labels
            st.video(annotated_video)
            # res=results[0].plot()
            # st.image(res)

if __name__ == '__main__':
    main()