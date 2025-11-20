# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=10, pretrained=True):
    # Use ResNet18 backbone
    model = models.resnet18(pretrained=pretrained)
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
