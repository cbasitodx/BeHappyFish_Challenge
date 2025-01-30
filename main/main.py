import torch
import shap
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

# Load the finetuned model
device : str = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18()  # Ajusta según tu arquitectura
model.load_state_dict(torch.load("modelo_resnet_finetuned.pt", map_location=device))
model.eval().to(device)

# 2️⃣ Definir preprocesamiento de imagen (según ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3️⃣ Cargar una imagen de prueba
img = Image.open("imagen_prueba.jpg")
img_tensor = transform(img).unsqueeze(0).to(device)

# 4️⃣ Definir la función de predicción para SHAP
def predict(x):
    tensor_x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Ajuste de dimensiones
    return model(tensor_x).detach().cpu().numpy()

# 5️⃣ Crear un explicador SHAP con GradientExplainer
background = torch.zeros((1, 3, 224, 224))  # Fondo de referencia
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(img_tensor.cpu().numpy())

# 6️⃣ Visualizar las explicaciones
shap.image_plot(shap_values, img_tensor.cpu().numpy())
