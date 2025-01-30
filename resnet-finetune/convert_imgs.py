import os
import shutil
import cv2

# Paths to original dataset
ORIG_DATASET = "images"  # Change to actual path
NEW_DATASET = "dataset"

# Class mapping
CLASS_NAMES = ["Eye", "Fish"]
EYE_CLASS_INDEX = CLASS_NAMES.index("Eye")

for split in ["train", "valid"]:
    os.makedirs(os.path.join(NEW_DATASET, split), exist_ok=True)


def process_split(split):
    """ Process train or valid split to extract only 'Eye' label images. """
    labels_path = os.path.join(ORIG_DATASET, split, "labels")
    images_path = os.path.join(ORIG_DATASET, split, "images")
    new_images_path = os.path.join(NEW_DATASET, split)

    for label_file in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_file)
        image_file = label_file.replace(".txt", ".jpg")  # Adjust if using PNG
        image_path = os.path.join(images_path, image_file)

        if not os.path.exists(image_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        eye_boxes = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id == EYE_CLASS_INDEX:
                eye_boxes.append([float(x) for x in parts[1:]])

        if eye_boxes:
            img = cv2.imread(image_path)
            h, w, _ = img.shape

            for i, (x_center, y_center, box_w, box_h) in enumerate(eye_boxes):
                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                eye_crop = img[y1:y2, x1:x2]

                if eye_crop.size > 0:
                    new_image_name = f"{os.path.splitext(image_file)[0]}_eye_{i}.jpg"
                    new_image_path = os.path.join(new_images_path, new_image_name)
                    cv2.imwrite(new_image_path, eye_crop)


# Define new subfolders
CATEGORIES = ["healthy", "sick"]

# Set a random seed for reproducibility


def split_images(split):
    """Splits images into 'healthy' and 'sick' folders within each split."""
    split_path = os.path.join(NEW_DATASET, split)
    images = [f for f in os.listdir(split_path) if f.lower().endswith(('.jpg', '.png'))]

    # Split into two equal parts
    mid_point = len(images) // 2
    healthy_images, sick_images = images[:mid_point], images[mid_point:]

    # Create subfolders
    for category in CATEGORIES:
        os.makedirs(os.path.join(split_path, category), exist_ok=True)

    # Move images
    for image in healthy_images:
        shutil.move(os.path.join(split_path, image), os.path.join(split_path, "healthy", image))

    for image in sick_images:
        shutil.move(os.path.join(split_path, image), os.path.join(split_path, "sick", image))


# Process train and valid splits
for split in ["train", "valid"]:
    process_split(split)

print("Dataset with only 'Eye' images has been created successfully.")
