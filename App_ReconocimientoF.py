import streamlit as st
from PIL import Image,ImageEnhance
import numpy as np
import os
import cv2 

#! cargaremos la imagen
@st.cache
def cargarImagen(img):
    imagen = Image.open(img)
    return imagen

#! llamamos al archivo xml face
rostroCascade = cv2.CascadeClassifier('esset/faceDetector.xml')

#! llamamos al archivo xml smile
smileCascade = cv2.CascadeClassifier('esset/haarcascade_smile.xml')

#! funcion para detectar rostro
def detectarRostro(mi_imagen):
    nuevaImagen = np.array(mi_imagen.convert('RGB'))
    imgScale = cv2.cvtColor(nuevaImagen, 1)
    gris = cv2.cvtColor(imgScale, cv2.COLOR_BGRA2GRAY)
    rostros = rostroCascade.detectMultiScale(gris,1.1,4)
    for (ejeX, ejeY, ancho, alto) in rostros:
        cv2.rectangle(imgScale, (ejeX,ejeY), (ejeX+ancho,ejeY+alto), (0,255,0), 2)
        
    return imgScale, rostros

#! funcion para detectar smile
def detectarSonrisa(mi_smile):
    nuevaSmile = np.array(mi_smile.convert('RGB'))
    imgScale = cv2.cvtColor(nuevaSmile, 1)
    gris = cv2.cvtColor(imgScale, cv2.COLOR_BGRA2GRAY)
    smile = smileCascade.detectMultiScale(gris,1.1,4)
    for (ejeX, ejeY, ancho, alto) in smile:
        cv2.rectangle(imgScale, (ejeX,ejeY), (ejeX+ancho,ejeY+alto), (0,255,0), 2)
        
    return imgScale, smile

#! funcion de CANNY
def codeCanny(mi_imagen):
    new_image = np.array(mi_imagen.convert("RGB"))
    imgCanny = cv2.cvtColor(new_image, 1)
    imgCanny = cv2.GaussianBlur(imgCanny, (11,11), 0)
    miCanny = cv2.Canny(imgCanny, 110,150)
    return miCanny

#! funcion para caricaturizar tu imagen
def cartoon(mi_imagen):
    nuevaCartoon = np.array(mi_imagen.convert('RGB'))
    imgCartoon = cv2.cvtColor(nuevaCartoon, 1)
    gris = cv2.cvtColor(imgCartoon, cv2.COLOR_BGRA2GRAY)
    gris = cv2.medianBlur(gris, 5)
    bordes = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,9)
    color = cv2.bilateralFilter(gris,9,300,300)
    caricatura = cv2.bitwise_and(color, color,mask=bordes)
    
    return caricatura

#! apps principal    
def main():
    st.set_page_config(page_title='Detector de imagenes',page_icon=':smile:', layout='wide')
    st.title('IA de imagenes')
    st.text('Probando la IA')
    actividades = ['Detectar','Acerca']
    eleccion = st.sidebar.selectbox('Selecciona una actividad', actividades)
    if eleccion == 'Detectar':
        st.subheader('Deteccion de rostro')
        file_img = st.file_uploader('Sube tu imagen',type=['jpg','png','gif','jpeg'])
        
        if file_img is not None:
            mi_imagen = Image.open(file_img)
            st.text('Original_image')
            st.image(mi_imagen)
            
        formatoImg = st.sidebar.radio('formato de imagen', ['Original', 'Escala de grises', 'Contraste', 'Brillante', 'Efecto Gausiano'])
        if formatoImg == 'Escala de grises':
            nuevaImagen = np.array(mi_imagen.convert('RGB'))
            img = cv2.cvtColor(nuevaImagen, 1)
            colorGris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # st.write(nuevaImagen)
            
            st.image(colorGris)
        elif formatoImg == 'Contraste':
            radioContraste = st.sidebar.slider('Contraste', 0.5,4.0)
            formatoContraste = ImageEnhance.Contrast(mi_imagen)
            fullcolor = formatoContraste.enhance(radioContraste)
            st.image(fullcolor)
            
        elif formatoImg == 'Brillante':
            blinkBlink = st.sidebar.slider('Brillante', 0.5,6.0)
            formatobrillante = ImageEnhance.Brightness(mi_imagen)
            fullstar = formatobrillante.enhance(blinkBlink)
            st.image(fullstar)
            
        elif formatoImg == 'Efecto Gausiano':
            nuevaImagen = np.array(mi_imagen.convert('RGB'))
            radioGasper = st.sidebar.slider('Efecto Gausiano', 0.5,4.5)
            imgGasper = cv2.cvtColor(nuevaImagen, 1)
            imgTransp = cv2.GaussianBlur(imgGasper,(11,11),radioGasper)
            st.image(imgTransp)
            
        animaciones = ['Faces', 'Smile', 'Canny', 'Cartoon']
        escogerAnimacion = st.sidebar.selectbox('Escoger la Animacion', animaciones)
        if st.button('Detectar'):
            
            if escogerAnimacion == 'Faces':
                resultadoImg, resultadoRostro = detectarRostro(mi_imagen)
                st.image(resultadoImg)
                st.success('{} Rostros encontrados '.format(len(resultadoRostro)))
            
            if escogerAnimacion == "Smile":
                resultadoImg, resultadoSmile = detectarSonrisa(mi_imagen)
                st.image(resultadoImg)
                st.success("{} Sonrisas encontradas".format(len(resultadoSmile)))

            if escogerAnimacion == "Canny":
                resultadoImg = codeCanny(mi_imagen)
                st.image(resultadoImg)

            if escogerAnimacion == "Cartoon":
                resultadoImg = cartoon(mi_imagen)
                st.image(resultadoImg)    
                
                
                
    elif eleccion == 'Acerca':
        st.subheader('Acerca de la imagen')
        

        
    
# if __name__=='__main__':
main()
