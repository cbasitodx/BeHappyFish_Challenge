# Python script to get the weight

import cv2
import os
import time
import numpy as np
import easyocr 
from ultralytics import YOLO
import shutil

_DEVICE = 'cuda'
_YOLO_MODEL_PATH = "model/classification/finetuned_numeros.pt"


def read_weight(image_path : str) -> float:

    image = cv2.imread(image_path)
    image_name = image_path.split('/')
    image_name = image_name[image_name.__len__()-1]




    number_detector : YOLO = YOLO(_YOLO_MODEL_PATH)
    number_detector.to(_DEVICE)



    if os.path.exists("runs/weight/detect/predict"):  # Check if directory exists
        shutil.rmtree("runs/weight/detect/predict")  # Delete it

    # Extract the fish eye and the fish
    number_detector(source = image_path, conf = 0.4, save = True, save_txt = True, project = "runs/weight/detect", name = "predict")

    label_path = "runs/weight/detect/predict/labels/" + image_name.replace(".png",".txt").replace(".jpg",".txt")
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
    number_box = number_box[:,:-20]

    min_val, max_val = np.percentile(number_box, (20, 90))  # Equivalent to "-contrast-stretch 10%x10%"
    img = np.clip((number_box - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)


    white = img > 80

    img[white] = 255


    # Initialize EasyOCR reader (English digits)
    reader = easyocr.Reader(['en'], gpu=True)

    results : str = "".join(reader.readtext(img, detail=0, allowlist="0123456789"))
    results2 : str = "".join(reader.readtext(number_box, detail=0, allowlist="0123456789"))
    print(results)
    print("Detected Digits:", "".join(results))
    print("Detected2: " + "".join(results2))

    # usar los dos resultados para sacar uno favorito
    res1 = int(results)/100
    res2 = int(results2)/100

    if(np.abs(1000 - res1) < np.abs(1000 - res2)):
        return res1
    else:
        return res2


    
            

    


    
                


