import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shap
from PIL import Image

# Hyper-parameters
MODE = 'train'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 100
LEARNING_RATE = 0.001
ROOTDIR = r'C:\Users\joshu\Documents\ArtificialIntelligence\BirdsCNN'

class BirdsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 0]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(6, 6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)   
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EntropyClassifier(nn.Module):
    def __init__(self):
        super(EntropyClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(6, 6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)   
        self.fc3 = nn.Linear(84, 2)

    def entropy(self, x):
        _x = x
        logx = torch.log(_x)
        out = _x * logx
        out = torch.sum(out, 1)
        out = out[:, None]
        return -out

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.entropy(x)
        return x


def train(model, train_loader, optimizer, criterion, path):
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            print(scores)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Train Epoch: ', epoch, 'Loss: ', np.mean(losses))
    torch.save(model.state_dict(), path)


#Create data loaders
dataset = BirdsDataset(csv_file='BUSHvsWILDTurkey.csv', root_dir=ROOTDIR, transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [300, 32])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)


model = CNNClassifier().to(DEVICE)
entropyModel = EntropyClassifier().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

if MODE == 'train':
    train(model, train_loader, optimizer, criterion, ROOTDIR + r'\model.pth')

if MODE =='test':
    model.load_state_dict(torch.load(ROOTDIR + r'\model.pth'))
    model.eval()
    image = Image.open(ROOTDIR + r'\test1.jpg')
    convert = transforms.ToTensor()
    image = convert(image).to(DEVICE)

if MODE == 'explain':
    entropyModel.load_state_dict(torch.load(ROOTDIR + r'\model.pth'))
    entropyModel.eval()

    image = Image.open(ROOTDIR + r'\BushTurkey.jpg')
    convert = transforms.ToTensor()
    image = convert(image).to(DEVICE)

    batch = next(iter(test_loader))
    images, _ = batch

    background = images[0:50].to(DEVICE)    

    explainer = shap.DeepExplainer(entropyModel, background)



