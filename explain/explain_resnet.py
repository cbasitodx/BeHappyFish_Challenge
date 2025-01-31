import torch
import shap
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

def explain(model : models.RegNet, not_transformed_image : Image, saving_path : str) -> None:
    device : str           = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Put the model in eval mode
    model.eval().to(device)

    # Create a transformation for the input images (essentially, it puts them in a particular size image and normalizes them)
    transform : transforms.Compose = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transform the image with the transformation
    img_tensor   : torch.Tensor = transform(not_transformed_image).unsqueeze(0).to(device)

    # Create a GradientExplainer object from the SHAP library
    background  : torch.Tensor           = torch.zeros((1, 3, 224, 224)).to(device)  # Reference value 
    explainer   : shap.GradientExplainer = shap.GradientExplainer(model, background)
    shap_values : np.ndarray             = explainer.shap_values(img_tensor.to(device))

    # Generate the visualization of the explanation
    shap.image_plot(shap_values.squeeze(0), img_tensor.squeeze(0).cpu().numpy(), show=False)
    plt.savefig(saving_path)

if __name__=="__main__":
    img = Image.open("/home/seby/Dev/BeHappyFish_Challenge/data/classification/train/healthy/ZHAW-Biocam_00_20240325093908_jpg.rf.290618dfca86b3aa7ba4d333ff49c4c0_eye_0.jpg")
    
    # Load the finetuned model
    device : str = "cuda" if torch.cuda.is_available() else "cpu"
    model        = models.resnet18(pretrained=True)

    # Change the final layer (fc) according to the number of classes in the fish eye illness dataset
    num_classes = 2  # Ajusta a tu número de clases
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load("./trained_model/classification/fine_tuned_best_model.pt", map_location=device))
    model.eval().to(device)

    explain(model, img, "./results/xxxx.png")