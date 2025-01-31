# Python script to get the weight

import cv2
import os
import time
import numpy as np
import pytesseract
import subprocess
import easyocr 
from ultralytics import YOLO

_DEVICE = 'cuda'
_YOLO_MODEL_PATH = "model/classification/finetuned_numeros.pt"



image = cv2.imread("Dataset/LINDA/EelisaHackathonDatasetsLinda/FishDiseaseZHAW/OverUnderWeight/ZHAW Biocam_00_20240325094126.jpg")

images = [f for f in os.listdir("Dataset/LINDA/EelisaHackathonDatasetsLinda/FishDiseaseZHAW/OverUnderWeight")]

for image_name in images:
    image_path = "Dataset/LINDA/EelisaHackathonDatasetsLinda/FishDiseaseZHAW/OverUnderWeight/" + image_name
    image = cv2.imread(image_path)
    image_old = image




    number_detector : YOLO = YOLO(_YOLO_MODEL_PATH)
    number_detector.to(_DEVICE)

    # Extract the fish eye and the fish
    number_detector(source = image_path, conf = 0.4, save = True, save_txt = True, project = "runs/detect", name = "predict")
    bounding_box_img = cv2.imread("runs/detect/predict/" + image_name)

    label_path = "runs/detect/predict/labels/" + image_name.replace(".png",".txt").replace(".jpg",".txt")
    datos = ""
    with open(label_path, 'r') as file:
        lineas = file.readlines()
        for linea in lineas:
            datos = linea.strip().split()


    total_img_width = image[0].__len__()
    total_img_height = image.__len__()


    box_width : int = int(total_img_width * float(datos[3]))
    box_height : int = int(total_img_height * float(datos[4]))

    box_x : int = int(total_img_width * float(datos[1]))
    box_y : int = int(total_img_height * float(datos[2]))



    #print(box_width)

    number_box = image[box_y-int(box_height/2):box_y+int(box_height/2),box_x-int(box_width/2):box_x+int(box_width/2)]

    number_box = cv2.cvtColor(number_box,cv2.COLOR_BGR2GRAY)

    min_val, max_val = np.percentile(number_box, (5, 95))  # Equivalent to "-contrast-stretch 10%x10%"
    img = np.clip((number_box - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
    min_val, max_val = np.percentile(img, (5, 95))
    img = np.clip((img - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)


    white = img > 80

    img[white] = 255

    cv2.imshow('bbox',img)
    cv2.imshow('old',number_box)




    #cv2.imshow('Cropped Yellow Rectangle', image)
    cv2.namedWindow('bounding box', cv2.WINDOW_NORMAL)
    cv2.imshow('bounding box', bounding_box_img)
    cv2.resizeWindow('bounding box', 500, 500)
    cv2.moveWindow('bounding box', 500, 500)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
            

    


    
                


