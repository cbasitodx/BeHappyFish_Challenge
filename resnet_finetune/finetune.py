import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from fine_tuning_config_file import *

import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataset(data_dir, batch_size, shuffle=True):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=4) for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, total, len(train_loader.dataset),
        100. * total / len(train_loader.dataset), train_loss / len(train_loader)))
    return train_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def main(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    dataloaders, dataset_sizes, class_names = get_dataset(data_dir, batch_size)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    scheduler = exp_lr_scheduler(optimizer, num_epochs)
    for epoch in range(num_epochs):
        train_loss = train(model, device, dataloaders['train'], optimizer, epoch)
        test_loss = test(model, device, dataloaders['valid'])

        if test_loss < best_acc:
            best_acc = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
