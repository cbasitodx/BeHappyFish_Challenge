import cv2
import numpy as np

# Load image
image = cv2.imread("Dataset/LINDA/EelisaHackathonDatasetsLinda/FishDiseaseZHAW/OverUnderWeight/ZHAW Biocam_00_20240325122615.jpg")  # Replace with your image file

x_start, y_start, x_end, y_end = 300,1050,1900,1900

image = image[y_start:y_end, x_start:x_end]

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the yellow color range in HSV
lower_yellow = np.array([20, 100, 100])  # Adjust if needed
upper_yellow = np.array([35, 255, 255])  # Adjust if needed

# Create a mask for yellow color
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Apply morphological operations to clean small noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)  # Adjust epsilon for more or less precision
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the contour has four vertices (a rectangle)
    if len(approx) == 4:
        # Calculate the aspect ratio of the bounding box
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        print(w)
        print(h)
        # Define an aspect ratio threshold for identifying a rectangle (may need adjustment)
        if w > 700 and w < 10000 and h > 100 and h < 250:
            # Draw the rectangle on the image
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)  # Draw green contour

            # Crop the detected rectangle from the image
            cropped_image = image[y:y+h, x:x+w]
            
            # Display the cropped image of the detected yellow rectangle
            cv2.imshow('Cropped Yellow Rectangle', cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

cv2.imshow('Cropped Yellow Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()





# # Convert the image to HSV color space
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define the yellow color range in HSV
#     lower_yellow = np.array([20, 100, 100])  # Adjust if needed
#     upper_yellow = np.array([35, 255, 255])  # Adjust if needed

#     # Create a mask for yellow color
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#     # Apply morphological operations to clean small noise
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     drawn : bool = False
#     for contour in contours:
#         # Approximate the contour to a polygon
#         epsilon = 0.04 * cv2.arcLength(contour, True)  # Adjust epsilon for more or less precision
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         # Check if the contour has four vertices (a rectangle)
#         if len(approx) == 4:
#             # Calculate the aspect ratio of the bounding box
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h

#             # Define an aspect ratio threshold for identifying a rectangle (may need adjustment)
#             if w > 700 and w < 10000 and h > 100 and h < 250:
#                 # Draw the rectangle on the image
#                 cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)  # Draw green contour

#                 # Crop the detected rectangle from the image
#                 cropped_image = image[y:y+h, x:x+w]
#                 drawn = True
                
#                 # Display the cropped image of the detected yellow rectangle
#                 # cv2.imshow('Cropped Yellow Rectangle', cropped_image)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
    
#     if( not drawn ):
#         print("ERROR: RECTANGULO NO RECONOCIDO")
#         # Display the cropped image of the detected yellow rectangle
#         cv2.imshow('Cropped Yellow Rectangle', image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()