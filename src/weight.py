# Python script to get the weight

import cv2
import os
import time
import numpy as np
import pytesseract
import subprocess
import easyocr 


image = cv2.imread("Dataset/LINDA/EelisaHackathonDatasetsLinda/FishDiseaseZHAW/OverUnderWeight/ZHAW Biocam_00_20240325094126.jpg")

images = [f for f in os.listdir("Dataset/LINDA/EelisaHackathonDatasetsLinda/FishDiseaseZHAW/OverUnderWeight")]

for image_name in images:
    image_path = "Dataset/LINDA/EelisaHackathonDatasetsLinda/FishDiseaseZHAW/OverUnderWeight/" + image_name
    image = cv2.imread(image_path)
    print(image_name)

    x_start, y_start, x_end, y_end = 900,1650,1600,1900

    image = image[y_start:y_end, x_start:x_end]
    imagen_old = image

    roi = image[:100, 100:x_end-x_start-100]  # Region of Interest (ROI) - first 100 rows

    # Convert ROI to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define HSV range for yellow
    lower_yellow = np.array([20, 100, 100])  # Lower bound of yellow
    upper_yellow = np.array([35, 255, 255])  # Upper bound of yellow

    # Create a mask for yellow pixels
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Check if any yellow pixels are present
    if np.any(mask > 50):
        # Find the last row with yellow (scan from bottom)
        last_yellow_row = np.where(mask.any(axis=1))[0]
        last_yellow_row = last_yellow_row[-1] if len(last_yellow_row) > 0 else None

        image = image[last_yellow_row:,:]


    
    
        # Apply contrast adjustment
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0.5)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Preprocess: Apply adaptive threshold to deal with varying background
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)

    # Optionally, blur to reduce noise
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Find contours to detect the region of interest (ROI)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # If necessary, find the rotated bounding box and rotate the image

    smallestw = 1000000000
    smallesth = 1000000000
    xglobal = 0
    yglobal = 0
    # for contour in contours:
    #     # Get bounding box
    #     x, y, w, h = cv2.boundingRect(contour)

    #     if(w > 500 and h > 100) and w < smallestw and h < smallesth:
    #         smallesth = h
    #         smallestw = w
    #         xglobal = x
    #         yglobal = y
    
    # if(smallesth < 100000):
    #     image = image[yglobal:yglobal+smallesth,xglobal:xglobal+smallestw]


    # Apply contrast stretching (normalize intensity)
    mask = image[:,:,2] > 100

    image[mask] = [255,255,255]

    min_val, max_val = np.percentile(image, (10, 90))  # Equivalent to "-contrast-stretch 10%x10%"
    img = np.clip((image - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)


    # Apply binary thresholding (equivalent to "-threshold 50%")
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # 127 is 50% of 255


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    red = image[:,:,2]

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 10, 150, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        

        # Check if the polygon has 4 vertices (rectangle)
        if len(approx) == 4:
            # Get the bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Calculate the area of the rotated rectangle
            width, height = rect[1]
            area = width * height
            
            # Check if the area is within the desired range
            if 25000 <= area <= 100000:

                x, y, w, h = cv2.boundingRect(box)
                x = max(0, x)  # Make sure x is not negative
                y = max(0, y)  # Make sure y is not negative
                w = min(w, image.shape[1] - x)  # Ensure the width does not go beyond the image width
                h = min(h, image.shape[0] - y)  # Ensure the height does not go beyond the image height

                
                # Draw the rotated rectangle
                image = image[y:y+h, x:x+w]

                
                
                
                break
    

    # Apply binary thresholding (equivalent to "-threshold 50%")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    white = gray == 255

    gray[~white] = 0






    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 10, 150, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        

        # Check if the polygon has 4 vertices (rectangle)
        if len(approx) == 4 or True:
            # Get the bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Calculate the area of the rotated rectangle
            width, height = rect[1]
            area = width * height
            
            # Check if the area is within the desired range
            if 35000 <= area <= 100000:
                
                print(area)
                x, y, w, h = cv2.boundingRect(box)
                x = max(0, x)  # Make sure x is not negative
                y = max(0, y)  # Make sure y is not negative
                w = min(w, image.shape[1] - x)  # Ensure the width does not go beyond the image width
                h = min(h, image.shape[0] - y)  # Ensure the height does not go beyond the image height

                #cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

                image = image[y:y+h, x:x+w]
                
                # Draw the rotated rectangle

    


    
    

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    white_mask = (gray == 255)

    # Get the minimum and maximum values (excluding white pixels)
    non_white_pixels = gray[~white_mask]  # Exclude white pixels from the calculation
    min_val, max_val = np.percentile(non_white_pixels, (10, 90))  # Contrast stretch percentiles


    # Perform contrast stretching on the non-white pixels
    image = np.clip((gray - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)

    # Set the white pixels back to 255 (to ensure they remain white)


    

    



    # Get the minimum and maximum values (excluding white pixels)
    non_white_pixels = gray[~white_mask]  # Exclude white pixels from the calculation
    min_val, max_val = np.percentile(non_white_pixels, (10, 90))  # Contrast stretch percentiles


    # Perform contrast stretching on the non-white pixels
    image = np.clip((gray - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)

    black = image < 40

    image[~black] = 255


    # # Initialize EasyOCR reader (English digits)
    reader = easyocr.Reader(['en'], gpu=True)

    results = reader.readtext(image, detail=0, allowlist="0123456789")
    print("Detected Digits:", "".join(results))



    


    cv2.imshow('Cropped Yellow Rectangle', image)
    cv2.imshow('old', imagen_old)
    cv2.moveWindow('old', 1000,500)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
            

    


    
                


