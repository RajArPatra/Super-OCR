import streamlit as st
from pdf2image import convert_from_path
import os
from PIL import Image
from main import predict
import numpy as np
import cv2
import pandas as pd
import base64
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata"'

def table_detection(img_path):
    img = cv2.imread(img_path)
    img_s= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    img_bin = cv2.bitwise_not(img_bin)
    kernel_length_v = (np.array(img_gray).shape[1])//200
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v)) 
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=5)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=5)

    kernel_length_h = (np.array(img_gray).shape[1])//100
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=5)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)
   
    contours, hierarchy = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    dict1={}
    lst1=['shipper','consignee','notify party']
    for i,c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if (w > 0 and h > 0) :
            count += 1
            cropped = img[y-3:y + h, x-3:x + w]
            
            txt=pytesseract.image_to_string(cropped, config=tessdata_dir_config).strip()
            for el in lst1:
              if el in  txt.split('\n')[0].lower():
                if cropped.shape[0]<img.shape[0]/20:
                  cropped = img[y-3:y+ h+int(img.shape[0]/14), x-3:x + w]
                  txt=pytesseract.image_to_string(Image.fromarray(cropped), config=tessdata_dir_config).strip()
                dict1.update({el:' '.join(txt.split('\n')[1:])})
              img_s=cv2.rectangle(img_s,(x-3,y-3),(x + w,y + h),(255,0,0),2)
    dict1=pd.DataFrame([dict1])
    return  dict1,img_s


st.title("TableNet with OCR Detection")
st.markdown("Hello There")

method = st.selectbox("Image or PDF", ['PDF', 'Image'])
if method == "PDF":
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
    if uploaded_file is not None:
        with open("selected.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        for _ in os.listdir("extracted_images"):
            os.remove(os.path.join("extracted_images", _))

        images = convert_from_path("selected.pdf")
        for i in range(len(images)):
            images[i].save('extracted_images/page'+'_'+ str(i+1) +'.jpg', 'JPEG')

        img_cols = st.beta_columns(len(images))
        for i in range(len(img_cols)):
            img_cols[i].subheader("page"+str(i+1))
            img_cols[i].image(Image.open("extracted_images/page_"+str(i+1)+".jpg"), use_column_width=True)

        selected_page = st.selectbox("Select the page", os.listdir("extracted_images"))

        image = Image.open('extracted_images/'+selected_page)
        st.image(image)

        selected_approach = st.selectbox("select approach",['Image Processing', 'TableNet approach'])
        if selected_approach == 'Image Processing':
            df, img = table_detection('extracted_images/'+selected_page)
            st.image(img, "processed image")
            st.dataframe(df)
            if not df.empty:
                csv = df.to_csv().encode()
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="out.csv" target="_blank">Download csv file</a>'
                st.markdown(href, unsafe_allow_html=True)
        if selected_approach == 'TableNet approach':
            out, tb, cl = predict('extracted_images/'+selected_page, 'best_model.ckpt')
            st.image(tb, "Table Mask")
            st.image(cl, "Column Mask")
            for i in range(len(out)):
                st.dataframe(out[i])
                csv = out[i].to_csv().encode()
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="out.csv" target="_blank">Download csv file</a>'
                st.markdown(href, unsafe_allow_html=True)
        


if method == "Image":
    st.write(method)
    uploaded_file = st.file_uploader("Choose an Image", type=['jpg','jpeg','png','bmp'])
    if uploaded_file is not None:
        with open("selected_img.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(Image.open('selected_img.jpg'), width=200)
        selected_approach = st.selectbox("select approach",['Image Processing', 'TableNet approach'])

        if selected_approach == 'Image Processing':
            df, img = table_detection('selected_img.jpg')
            st.image(img, "processed image")
            st.dataframe(df)
            if not df.empty:
                csv = df.to_csv().encode()
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="out.csv" target="_blank">Download csv file</a>'
                st.markdown(href, unsafe_allow_html=True)
        if selected_approach == 'TableNet approach':
            out, tb, cl = predict('selected_img.jpg', 'best_model.ckpt')
            st.image(tb, "Table Mask")
            st.image(cl, "Column Mask")
            for i in range(len(out)):
                st.dataframe(out[i])
                csv = out[i].to_csv().encode()
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="out.csv" target="_blank">Download csv file</a>'
                st.markdown(href, unsafe_allow_html=True)