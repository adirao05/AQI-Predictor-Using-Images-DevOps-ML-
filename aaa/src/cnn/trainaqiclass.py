import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from dataset import TRAQIDDataset
from model_cnn_class import CNNClassifier


CSV_PATH = r"C:\Users\salai_wciilqo\Downloads\traqid\TRAQID.csv"
IMAGE_DIR = r"E:\sem 6\aqi project\front_jpg"

BATCH_SIZE = 96
EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TRAQIDDataset(CSV_PATH,IMAGE_DIR,target="aqi_cat")

train_size = int(0.8*len(dataset))
val_size = int(0.1*len(dataset))
test_size = len(dataset)-train_size-val_size

train_dataset,val_dataset,test_dataset = random_split(dataset,[train_size,val_size,test_size])

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE)


model = CNNClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss {total_loss/len(train_loader):.4f}")


torch.save(model.state_dict(),"outputs/models/cnn_aqi_class.pth")