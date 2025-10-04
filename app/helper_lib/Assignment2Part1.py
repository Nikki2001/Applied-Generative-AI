import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

torch.manual_seed(42) 
np.random.seed(42)

# Check which GPU is available
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Using device: {device}')

BATCH_SIZE = 64

# Prepare the Data
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

torch.Size([3, 64, 64])
train_dataset[0][0].shape

first_img, first_label = train_dataset[0]
print("Label: ", first_label)
plt.imshow(first_img.permute(1,2,0).squeeze())
plt.show()

#np.str_('frog')
CLASSES = np.array(
    [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)
CLASSES[first_label]

print("Some individual pixel: ", train_dataset[54][0][1, 12, 13])
print("Corresponding Label: ", train_dataset[54][1])

_random_index = np.random.randint(len(train_dataset))
_img, _label = train_dataset[_random_index]
print("Label: ", _label, CLASSES[_label])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Length of the train dataset: ", len(train_dataset))
print("Length of list(train_loader): ", len(list(train_loader)))
print("Shape of the first element of list(train_loader)[0]: ", list(train_loader)[0][0].shape)

next_batch_images, next_batch_labels = next(iter(train_loader))
_first_img = next_batch_images[0] # retrieve the first image from the batch of 32
_first_label = next_batch_labels[0] # retrieve the first label from the batch of 32
plt.imshow(_first_img.permute(1, 2, 0)) # imshow requires the image to be in height x width x channels format
plt.show()
print("Label: ", CLASSES[_first_label])

# Parameters
NUM_CLASSES = 10
EPOCHS = 1

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 64x64x3 -> 64x64x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                  # halves H,W
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 32x32x16 -> 32x32x32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # After two pools: 64->32->16, so 32*16*16=8192, then 100 units
        self.fc2 = nn.Linear(100, 10)            # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # -> [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))     # -> [B, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16)             # Flatten
        x = F.relu(self.fc1(x))                  # 100 units
        x = self.fc2(x)                          # 10 classes
        return x

net = SimpleCNN()

model = SimpleCNN().to(device)
print(model)

from tqdm import tqdm

datalogs = []

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(EPOCHS):
    running_loss = 0.0
    running_correct, running_total = 0, 0

    model.train()
    train_loader_with_progress = tqdm(iterable=train_loader, ncols=120, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
        inputs = inputs.to(device)
        labels = labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # predicted = torch.argmax(outputs.data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log data for tracking
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        running_loss += loss.item()  

        if (batch_number % 100 == 99):
            train_loader_with_progress.set_postfix({'avg accuracy': f'{running_correct/running_total:.3f}', 'avg loss': f'{running_loss/(batch_number+1):.4f}'})

            datalogs.append({
                "epoch": epoch + batch_number / len(train_loader), 
                "train_loss": running_loss / (batch_number + 1),
                "train_accuracy": running_correct/running_total,
            })

    datalogs.append({
        "epoch": epoch + 1, 
        "train_loss": running_loss / len(train_loader),
        "train_accuracy": running_correct/running_total,
    })

    widget.src = losschart(datalogs)

print("Finished Training")
