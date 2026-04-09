import torch
import numpy as np

from torch.utils.data import DataLoader, random_split

from dataset import TRAQIDDataset
from model_mobilenet import MobileNetModel

from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():

    CSV_PATH = r"C:\Users\salai_wciilqo\Downloads\traqid\TRAQID.csv"
    IMAGE_DIR = r"E:\sem 6\aqi project\front_jpg"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", DEVICE)

    # ---------------------------
    # Load Dataset
    # ---------------------------

    dataset = TRAQIDDataset(
        csv_file=CSV_PATH,
        image_dir=IMAGE_DIR,
        target="PM2.5"
    )

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    test_loader = DataLoader(test_dataset, batch_size=32)

    # ---------------------------
    # Load Model
    # ---------------------------

    model = MobileNetModel().to(DEVICE)

    model.load_state_dict(
        torch.load("outputs/models/mobilenet_aqi.pth")
    )

    model.eval()

    preds = []
    targets = []

    # ---------------------------
    # Evaluation
    # ---------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images).squeeze().cpu().numpy()

            preds.extend(outputs)
            targets.extend(labels.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)

    print("\nMobileNetV2 Test Results")
    print("RMSE:", rmse)
    print("MAE:", mae)


if __name__ == "__main__":
    main()