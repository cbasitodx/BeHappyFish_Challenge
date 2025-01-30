import os
import cv2
import re

# Paths
YOLO_IMAGES_PATH = "images"  # Folder containing YOLO dataset images
YOLO_LABELS_PATH = "images"  # YOLO dataset labels are inside 'images/{split}/labels'
EYE_IMGS_PATH = "eye_imgs"  # Folder with healthy/sick classification
NEW_DATASET_PATH = "dataset"

# Class mapping (Eye comes before Fish)
CLASS_NAMES = ["Eye", "Fish"]
EYE_CLASS_INDEX = CLASS_NAMES.index("Eye")

# Healthy and sick folder mapping
CATEGORY_MAP = {
    "EyeHealthy": "healthy",
    "EyeIssue": "sick"
}

def get_base_name(yolo_filename):
    """Extracts the base name from a YOLO filename to match 'eye_imgs' format."""
    match = re.match(r"(ZHAW[-_]Biocam_\d+_\d+)_jpg\.rf\..+\.jpg", yolo_filename)
    if match:
        return match.group(1).replace("-", " ") + ".jpg"  # Convert dashes to spaces
    return None

def get_category(yolo_filename):
    """Determines if an image is 'healthy' or 'sick' based on 'eye_imgs' filenames."""
    base_name = get_base_name(yolo_filename)
    if base_name:
        for category, label in CATEGORY_MAP.items():
            if os.path.exists(os.path.join(EYE_IMGS_PATH, category, base_name)):
                return label
    return None  # Ignore images not found in 'eye_imgs'

def process_split(split):
    """Processes train or valid split, extracting only 'Eye' and classifying."""
    images_path = os.path.join(YOLO_IMAGES_PATH, split, "images")
    labels_path = os.path.join(YOLO_IMAGES_PATH, split, "labels")

    for image_file in os.listdir(images_path):
        if not image_file.lower().endswith(('.jpg', '.png')):
            continue

        category = get_category(image_file)
        if category is None:
            continue  # Skip images not in 'eye_imgs'

        label_file = image_file.replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(labels_path, label_file)

        if not os.path.exists(label_path):
            continue  # Skip images with no label file

        with open(label_path, "r") as f:
            lines = f.readlines()

        eye_boxes = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id == EYE_CLASS_INDEX:
                eye_boxes.append([float(x) for x in parts[1:]])

        if eye_boxes:
            img = cv2.imread(os.path.join(images_path, image_file))
            h, w, _ = img.shape

            for i, (x_center, y_center, box_w, box_h) in enumerate(eye_boxes):
                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                eye_crop = img[y1:y2, x1:x2]

                if eye_crop.size > 0:
                    new_folder = os.path.join(NEW_DATASET_PATH, split, category)
                    os.makedirs(new_folder, exist_ok=True)

                    new_image_name = f"{os.path.splitext(image_file)[0]}_eye_{i}.jpg"
                    new_image_path = os.path.join(new_folder, new_image_name)
                    cv2.imwrite(new_image_path, eye_crop)

# Process train and valid splits
for split in ["train", "valid"]:
    process_split(split)

print("Dataset has been categorized into 'healthy' and 'sick' with only 'Eye' images.")
