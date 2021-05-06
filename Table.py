
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract 

def table_detection(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray)
    #plt.show()
    #(thresh, img_bin) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (thresh, img_bin) = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    img_bin = cv2.bitwise_not(img_bin)
    #plt.imshow(img_bin )
    #cv2.imwrite('img.jpg',img_bin)
    #plt.show()
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
            
            txt=pytesseract.image_to_string(cropped).strip()
            for el in lst1:
              if el in  txt.split('\n')[0].lower():
                if cropped.shape[0]<img.shape[0]/20:
                  cropped = img[y-3:y+ h+int(img.shape[0]/14), x-3:x + w]
                  txt=pytesseract.image_to_string(cropped).strip()
                dict1.update({el:' '.join(txt.split('\n')[1:])})



     
	    #cv2.imwrite("table.jpg", table_segment)
	    #cv2.imwrite("img.jpg", img)
    dict1=pd.DataFrame([dict1])
    return  dict1
