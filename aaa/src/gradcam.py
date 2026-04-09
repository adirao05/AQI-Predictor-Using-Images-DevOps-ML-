import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

IMAGE_DIR = r"E:\sem 6\aqi project\front_jpg"

categories = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]

# load all images
all_images = os.listdir(IMAGE_DIR)

# --------------------------------
# function to avoid night images
# --------------------------------

def is_day_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    return brightness > 80


selected = []

while len(selected) < 24:

    img_name = random.choice(all_images)

    path = os.path.join(IMAGE_DIR,img_name)

    img = cv2.imread(path)

    if img is None:
        continue

    if is_day_image(img):

        selected.append(img)

# --------------------------------
# Plot Grid
# --------------------------------

fig, axes = plt.subplots(4,6, figsize=(18,10))

idx = 0

for r in range(4):

    for c in range(6):

        img = cv2.cvtColor(selected[idx], cv2.COLOR_BGR2RGB)

        axes[r,c].imshow(img)
        axes[r,c].axis("off")

        idx += 1

for i,title in enumerate(categories):

    axes[0,i].set_title(title, fontsize=16)

plt.tight_layout()

plt.savefig("outputs/plots/random_dataset_grid.png", dpi=300)

plt.show()