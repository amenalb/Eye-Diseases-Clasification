import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(script_dir, "dataset")

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(data_dir, transform=data_transform)

print(f'Liczba obrazów w zbiorze danych: {len(full_dataset)}')
print(f'Dostępne klasy: {full_dataset.classes}')

dimension_counts = defaultdict(int)
for image, label in full_dataset:
    dimensions = image.size()
    dimension_counts[dimensions] += 1
for dimensions, count in dimension_counts.items():
    print(f'Wymiary: {dimensions}, Liczba wystąpień: {count}')