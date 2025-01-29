import torch
from ultralytics import YOLO

print(torch.cuda.is_available())
if (torch.cuda.is_available()):
    cuda = torch.device('cuda')
    print(cuda)
else:
    print("No GPU available")

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11m-fish.pt")
model = model.to('cpu')

# Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
#results = model("C:\\Users\\capde\\Desktop\\IA\\YOLOv11\\peces.jpg", save = True, show = True)
results = model("C:\\Users\\capde\\Desktop\\IA\\YOLOv11\\Fish_video.mp4", save = True, show = True)
