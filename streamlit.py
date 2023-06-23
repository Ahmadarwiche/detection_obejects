import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import av
import cv2
import imghdr
import moviepy.editor as mpy
from utils import process_image, process_video
model = YOLO('models/best.pt')
st.set_page_config(page_title="Computer vision", page_icon="🖥️")

def main():
    st.title("🪴")

    # Add file uploader
    file = st.file_uploader("Upload Image or Video of your plan to detect a disease", type=["jpg", "jpeg", "png", "mp4", "mov"])

    if file is not None:
        file_extension = file.name.split(".")[-1].lower()
        if file_extension in ['jpg', 'jpeg', 'png']:
                # Process uploaded image
                img = Image.open(file)
                results= model(img)
                # results = model.predict(img, conf=0.25)

                # # Plot the results on the image
                res_plotted = results[0].plot(labels=True)
                st.image(res_plotted, caption='Predicted Image', use_column_width=True)


        elif file.type.startswith('video'):
            # Enregistrer le fichier téléchargé sur le disque
            with open('uploaded_video.mp4', 'wb') as f:
                f.write(file.getbuffer())

            # Charger la vidéo
            video = mpy.VideoFileClip('uploaded_video.mp4')

            # Reduire les fps de la vidéo
            video = video.set_fps(30)

            # Reduire la taille de la vidéo à 480p
            video = video.resize(height=480)

            # Obtenir le nombre total de frames dans la vidéo
            total_frames = int(video.fps * video.duration)

            # Créer une barre de progression
            progress_bar = st.progress(0)

            # Indiquer que le traitement est en cours
            with st.spinner('Traitement de la vidéo en cours...'):
                # Créer une liste pour stocker les images traitées
                processed_frames = []

                # Boucle de traitement de chaque image de la vidéo
                for i, frame in enumerate(video.iter_frames()):
                    # Convertir l'image en format PIL
                    image_pil = Image.fromarray(frame)

                    # Appliquer la fonction de traitement à chaque image
                    processed_frame = process_image(image_pil)

                    # Ajouter l'image traitée à la liste
                    processed_frames.append(processed_frame)

                    # Mettre à jour la barre de progression
                    progress_bar.progress(min((i + 1) / total_frames, 1.0))

            # Convertir la liste d'images traitées en un tableau numpy
            processed_frames_np = np.array(processed_frames)

            # Créer une vidéo à partir des images traitées
            output_video = mpy.ImageSequenceClip(list(processed_frames_np), fps=video.fps)

            # Ajouter le son
            output_video = output_video.set_audio(video.audio)

            # Enregistrer la vidéo traitée
            output_video.write_videofile("processed_video.mp4", codec='libx264')

            # Afficher la vidéo traitée
            st.video("processed_video.mp4")


if __name__ == '__main__':
    main()