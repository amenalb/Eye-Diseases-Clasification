import torch
import os
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from torchvision.models import ResNet18_Weights

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

# dimension_counts = defaultdict(int)
# for image, label in full_dataset:
#     dimensions = image.size()
#     dimension_counts[dimensions] += 1
# for dimensions, count in dimension_counts.items():
#     print(f'Wymiary: {dimensions}, Liczba wystąpień: {count}')


train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print("epoch_start", end="")
    for inputs, labels in train_loader:
        print(".", end="")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f'Epoka {epoch+1}/{epochs}, Strata: {running_loss/len(train_loader):.4f}, Dokładność: {100 * correct / total:.2f}%')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f'Dokładność walidacji: {100 * correct / total:.2f}%')
