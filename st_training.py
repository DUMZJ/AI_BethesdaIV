import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from swin_transformer import SwinTransformer

# Define hyperparameter
batch_size = 32
num_classes = 3
num_epochs = 10
learning_rate = 0.001

# Defining Datasets and Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root='path/to/train/data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the model, optimizer, and loss function
model = SwinTransformer()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Started training
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward propagation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Optimization and backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# Save the result
torch.save(model.state_dict(), 'swin_transformer.ckpt')
