import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset():
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = "../data/classification/valid"
    valid_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    return valid_loader


def evaluate(test_loader, model, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18().to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('../trained_model/classification/fine_tuned_best_model.pt', weights_only=True))

    valid_loader = load_dataset()

    y_true, y_pred = evaluate(valid_loader, model, device)
    class_names = ["sick", "healthy"]

    print(classification_report(y_true, y_pred, target_names=class_names, digits=6))
    plot_confusion_matrix(y_true, y_pred, class_names)


if __name__ == '__main__':
    main()
