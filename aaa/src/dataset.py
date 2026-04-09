import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class TRAQIDDataset(Dataset):

    def __init__(self, csv_file, image_dir, target="PM2.5"):

        self.label_map = {
            "Good":0,
            "Satisfactory":1,
            "Moderate":2,
            "Poor":3,
            "Very Poor":4,
             "Severe":5
        }
        self.image_dir = image_dir
        self.target = target

        data = pd.read_csv(csv_file)

        valid_rows = []

        for _, row in data.iterrows():

            image_id = str(row["Image"])

            img_path = os.path.join(image_dir, image_id + ".jpg")

            if os.path.exists(img_path):
                valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

        print("Total samples after cleaning:", len(self.data))

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_id = str(row["Image"])

        img_path = os.path.join(self.image_dir, image_id + ".jpg")

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(row[self.target], dtype=torch.float32)

        return image, label