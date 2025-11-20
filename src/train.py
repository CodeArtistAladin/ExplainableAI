# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_dataloaders
from model import get_model
import os

def train(epochs=3, batch_size=128, save_path='models/resnet_cifar10.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, classes = get_dataloaders(batch_size=batch_size)
    model = get_model(num_classes=len(classes), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=running/ (pbar.n+1))

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f'Epoch {epoch+1} test acc: {acc:.4f}')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print('Model saved to', save_path)
    return model, classes

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == '__main__':
    train(epochs=3, batch_size=128)
