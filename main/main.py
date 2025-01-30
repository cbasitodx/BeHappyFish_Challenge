import torch
import shap
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

# Load the finetuned model
device : str           = "cuda" if torch.cuda.is_available() else "cpu"
model  : models.ResNet = models.resnet18(pretrained=True)

# Change the final layer (fc) according to the number of classes in the fish eye illness dataset
num_classes : int  = 2
model.fc           = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("./trained_model/classification/fine_tuned_best_model.pt", map_location=device))
model.eval().to(device)

# Create a transformation for the input images (essentially, it puts them in a particular size image and normalizes them)
transform : transforms.Compose = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load a random test image
img_path_healthy : str = "./data/classification/valid/healthy"
img_path_sick    : str = "./data/classification/valid/sick"

images : list[str] = []
images.extend([os.path.join(img_path_healthy, f) for f in os.listdir(img_path_healthy) if os.path.isfile(os.path.join(img_path_healthy, f))])
images.extend([os.path.join(img_path_sick, f) for f in os.listdir(img_path_sick) if os.path.isfile(os.path.join(img_path_sick, f))])

chosen_image : str          = random.choice(images)
img          : Image        = Image.open(chosen_image)
img_tensor   : torch.Tensor = transform(img).unsqueeze(0).to(device)

# Function that will be used by shap to perform a prediction
def predict(x : torch.Tensor) -> np.ndarray:
    tensor_x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Ajuste de dimensiones
    return model(tensor_x).detach().cpu().numpy()

# Create a GradientExplainer object from the SHAP library
background  : torch.Tensor           = torch.zeros((1, 3, 224, 224)).to(device)  # Reference value 
explainer   : shap.GradientExplainer = shap.GradientExplainer(model, background)
shap_values : np.ndarray             = explainer.shap_values(img_tensor.to(device))


# Visualize the explanations
shap.image_plot(shap_values.squeeze(0), img_tensor.squeeze(0).cpu().numpy(), show=False)

filename : str = chosen_image.split("/")[-1]
plt.savefig(f"./results/{filename}")