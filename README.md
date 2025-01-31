# BeHappyFish Challenge

Welcome to the **BeHappyFish Challenge** repository! This project presents an optimized solution for detecting fish weight categories and eye health using deep learning models. Our approach prioritizes industrial applicability, efficiency, and scalability over purely academic or visually appealing methods.

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Advantages](#key-advantages)
- [License](#license)
- [Contact](#contact)

## About the Project

This repository contains a deep learning-based system designed for the **BeHappyFish Challenge**. Our solution is built with real-world industrial applications in mind, focusing on performance, reliability, and cost-effectiveness. The main goal is to provide a **scalable, efficient, and practical solution** for detecting fish weight categories (overweight, underweight, normal weight) and identifying whether a fish has a **healthy or sick eye**. While we currently do not classify specific diseases, we determine overall eye health.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- [Other dependencies listed in `requirements.txt`]

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cbasitodx/BeHappyFish_Challenge.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BeHappyFish_Challenge
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

[Provide specific details on how to run the model, including dataset requirements, inference steps, and example commands.]

## Project Structure

```
BeHappyFish_Challenge/
├── docs/
├── explain/
├── model/
├── overweight_detection/
├── resnet_finetune/
├── resnet_validation/
├── trained_model/
├── web/
├── .gitignore
├── LICENSE
├── README.md
├── main.py
└── requirements.txt
```

- `docs/`: Documentation for the project.
- `explain/`: Explanation of the methodology and results.
- `model/`: Core model architecture and training scripts.
- `overweight_detection/`: Implementation for detecting fish weight categories.
- `resnet_finetune/`: Fine-tuning scripts for ResNet-based models.
- `resnet_validation/`: Model validation and performance evaluation.
- `trained_model/`: Pretrained model weights.
- `web/`: Web-based interface for model deployment.
- `main.py`: Main execution script.
- `requirements.txt`: List of dependencies.

## Key Advantages

Our approach is superior due to several critical factors:

### **1. Vertical Scalability**

- Our model can be extended to detect more types of fish diseases beyond just eye health assessment.
- This allows for continuous improvement without requiring a complete system overhaul.

### **2. Horizontal Scalability**

- Designed to handle a **massive data volume** efficiently.
- The system supports seamless model switching, ensuring adaptability as new models become available.
- The architecture is built for **high throughput processing**, making it ideal for large-scale industrial applications.

### **3. Real Industrial Application**

- Unlike other solutions, our system was built **with industry needs in mind**.
- Every aspect of the pipeline is optimized to the very last detail for **maximum efficiency and practicality**.

### **4. Bounding Boxes Over Segmentation**

- **Segmenting fish images may look visually appealing, but it is unnecessary for the actual problem.**
- Masks are a **more complex data structure** than simple bounding boxes, leading to unnecessary computational overhead.
- **Bounding boxes** provide all the necessary information while being **simpler, faster, and more applicable** in real-world scenarios.

### **5. Finetuned Models with Real Data**

- Unlike generic models trained on unrelated datasets, our models are **finetuned with real fish images in a controlled environment**.
- Extensive validation has demonstrated **high accuracy and reliability** in real-world conditions.

### **6. Cost-Effective Solution**

- Our approach is designed to be **affordable**, ensuring that it can be widely adopted in the industry.
- Optimized computation reduces hardware costs while maintaining high performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact at [acm.fi.upm@gmail.com].
