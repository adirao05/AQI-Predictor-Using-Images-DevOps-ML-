import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from dataset import TRAQIDDataset
from model_cnn import CNNModel

import os


def main():

    # ---------------------------
    # Configuration
    # ---------------------------

    CSV_PATH = r"C:\Users\salai_wciilqo\Downloads\traqid\TRAQID.csv"
    IMAGE_DIR = r"E:\sem 6\aqi project\front_jpg"

    TARGET = "aqi"

    BATCH_SIZE = 110
    EPOCHS = 20
    LR = 1e-3

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", DEVICE)

    # ---------------------------
    # Load Dataset
    # ---------------------------

    dataset = TRAQIDDataset(
        csv_file=CSV_PATH,
        image_dir=IMAGE_DIR,
        target=TARGET
    )

    print("Dataset size:", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # ---------------------------
    # DataLoaders
    # ---------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ---------------------------
    # Model (CNN)
    # ---------------------------

    model = CNNModel().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---------------------------
    # Training Loop
    # ---------------------------

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images).squeeze()

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)}")

        epoch_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} completed | Loss: {epoch_loss:.4f}")

    # ---------------------------
    # Save Model
    # ---------------------------

    os.makedirs("outputs/models", exist_ok=True)

    torch.save(
        model.state_dict(),
        "outputs/models/cnn_aqi.pth"
    )

    print("CNN model saved!")


if __name__ == "__main__":
    main()