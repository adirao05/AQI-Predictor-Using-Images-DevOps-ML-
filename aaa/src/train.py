import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TRAQIDDataset
from model_cnn import CNNModel

CSV_PATH = r"C:\Users\salai_wciilqo\Downloads\traqid\TRAQID.csv"
IMAGE_DIR = r"E:\sem 6\aqi project\front_jpg"

BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TRAQIDDataset(csv_file=CSV_PATH, image_dir=IMAGE_DIR, target="PM2.5")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNNModel().to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images).squeeze()

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "cnn_pm25.pth")

print("Model saved")