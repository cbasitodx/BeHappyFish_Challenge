# ************* YOLO v8 FINETUNING MODEL FOR EYE DETECTION ************* #
import torch
from ultralytics import YOLO
from IPython.display import display
from IPython import display

# Choose the device for training the model
device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clean the output
display.clear_output()

# Hyperparameters & parameters
EPOCHS           : int = 10
BATCH_SIZE       : int = 8 
IMG_SIZE         : int = 640
     
CONFIG_NAME      : str = "detection_config.yaml"
CONFIG_PATH      : str = f"./data/detection/{CONFIG_NAME}"
MODEL_NAME       : str = "yolov8n.pt"
MODEL_PATH       : str = f"./model/detection/{MODEL_NAME}"
SAVE_PATH        : str = "./trained_model/detection"
TUNED_MODEL_NAME : str = f"{MODEL_NAME[:-3]}_trained.pt"

# Load (or download) the pre-trained model
model : YOLO = YOLO(MODEL_PATH)

# Empty CUDA cache
if(device == "cuda") : torch.cuda.empty_cache()

# Train the model
model.train(data=CONFIG_PATH, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, device=device, project=SAVE_PATH, name=TUNED_MODEL_NAME)