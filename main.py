import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from ultralytics import YOLO

from explain.explain_resnet import explain
from overweight_detection.weight import read_weight
from overweight_detection.under_over_weight import under_over_weight

UNDERWEIGHT        : int = 0
NORMAL             : int = 1
OVERWEIGHT         : int = 2

EYE                : int = 0
FISH               : int = 1

HEALTHY            : int = 0
SICK               : int = 1

_YOLO_MODEL_PATH   : str = "./trained_model/detection/yolov8n.pt_trained.pt/weights/best.pt"
_RESNET_MODEL_PATH : str = "./trained_model/classification/fine_tuned_best_model.pt" 

_DEVICE : str = "cuda" if torch.cuda.is_available() else "cpu"

def main(img_path : str, 
         yolo_saving_path : str,
         shap_saving_path : str, 
         yolo_weight_saving_path : str,
         detect_eye_disease : bool, 
         detect_overweight : bool, 
         explain_model : bool) -> dict[int, bool]:
    
    # Dictionary to be returned
    # "weight_classification" : (0 or 1 or 2)
    # "eye_classification"    : (True if healthy, False if not)}
    results_dict : dict[int, bool, str, str] = dict()

    # Start by loading the image
    img : Image = Image.open(img_path)

    # Load the models
    fish_and_eye_detector : YOLO = YOLO(_YOLO_MODEL_PATH)
    fish_and_eye_detector.to(_DEVICE)

    eye_disease_classifier : models.ResNet

    # Extract the fish eye and the fish
    bounding_boxes_img = fish_and_eye_detector(source = img_path, conf = 0.4, save = False)
    
    fish_and_eye_images : list[np.ndarray] = [0]*2
    
    for result in bounding_boxes_img:
        img_with_boxes = result.plot()
        img_with_boxes_pil = Image.fromarray(img_with_boxes)
        img_with_boxes_pil.save(yolo_saving_path)
    
        # Get the images inside the bounding boxes
        for (x1, y1, x2, y2, conf, cls) in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(cls)

            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img = Image.fromarray(np.array(cropped_img))

            if(class_id == EYE):
                fish_and_eye_images[EYE] = cropped_img
            
            elif(class_id == FISH):
                fish_and_eye_images[FISH] = cropped_img

    if(detect_eye_disease):
        eye_disease_classifier = models.resnet18(pretrained=True)

        # Change the final layer (fc) according to the number of classes in the fish eye illness dataset
        num_classes : int = 2
        eye_disease_classifier.fc = torch.nn.Linear(eye_disease_classifier.fc.in_features, num_classes)

        eye_disease_classifier.load_state_dict(torch.load(_RESNET_MODEL_PATH, map_location=_DEVICE))
        eye_disease_classifier.eval().to(_DEVICE)

        transform : transforms.Compose = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

        # Transform the image with the transformation
        eye_img : Image = fish_and_eye_images[EYE]
        eye_img_tensor : torch.Tensor = transform(eye_img).to(_DEVICE)

        # Obtain the results
        healthy_eye : bool = True if torch.argmax(eye_disease_classifier(eye_img_tensor)) == HEALTHY else False
        print(eye_disease_classifier(eye_img_tensor))

        results_dict["eye_classification"] = healthy_eye

        # Explain if needed
        if(explain_model):
            explain(model=eye_disease_classifier, not_transformed_image=eye_img, saving_path=shap_saving_path)

    if(detect_overweight):
        # Start by getting the weight of the fish
        fish_weight : float = read_weight(img_path, yolo_weight_saving_path)

        # Now, classify it as underweight, normal or overweight
        fish_img : Image = fish_and_eye_images[FISH]

        WIDTH_IMG : int
        width_box : int

        WIDTH_IMG, _ = img.size  
        width_box, _ = fish_img.size

        weight_classification : int = under_over_weight(width_box, WIDTH_IMG, fish_weight)

        results_dict["weight_classification"] = weight_classification

    return results_dict

if __name__=="__main__":
    # HEALTHY
    healthy = "/home/seby/Dev/BeHappyFish_Challenge/data/detection/train/images/ZHAW-Biocam_00_20240325093908_jpg.rf.290618dfca86b3aa7ba4d333ff49c4c0.jpg"
    
    # SICK
    sick = "/home/seby/Dev/BeHappyFish_Challenge/data/detection/train/images/ZHAW-Biocam_00_20240325112402_jpg.rf.4c5c31cfa194825beab8d76aae65bd81.jpg"

    x = main(healthy, 
         "./results/yolo.png",
         "./results/shap.png",
         "./results/",
         True,
         True,
         True)
    
    print(x)