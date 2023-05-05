import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from resnet import resnet50

# define the parameters
batch_size = 32
num_classes = 3
num_epochs = 10
learning_rate = 0.001


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root='path/to/train/data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = resnet50(num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        outputs = model(images)
        loss = criterion(outputs, labels)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))


torch.save(model.state_dict(), 'resnet50.ckpt')
