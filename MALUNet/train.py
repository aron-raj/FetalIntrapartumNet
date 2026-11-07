import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# ================================
# 1. DEVICE CHECK
# ================================
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# 2. DATASET (MULTI-CLASS FIXED)
# ================================
class UltrasoundDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path)

        img = cv2.resize(img, (256,256)) / 255.0
        mask = cv2.resize(mask, (256,256))

        # 🔥 Convert RGB mask → class labels
        label_mask = np.zeros((256,256), dtype=np.uint8)

        # Green → FH (class 1)
        label_mask[(mask[:,:,1] > 150)] = 1

        # Red → PS (class 2)
        label_mask[(mask[:,:,2] > 150)] = 2

        img = torch.tensor(img).unsqueeze(0).float()
        mask = torch.tensor(label_mask).long()

        return img, mask


# ================================
# 3. MALUNET ARCHITECTURE
# ================================
from model import MALUNet


# ================================
# 4. PATHS
# ================================
img_dir = r"C:\Users\aronr\ultra\dataset\JNU-IFM\dataset\images"
mask_dir = r"C:\Users\aronr\ultra\dataset\JNU-IFM\dataset\masks"


# ================================
# 5. LOAD DATA
# ================================
dataset = UltrasoundDataset(img_dir, mask_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Dataset size:", len(dataset))


# ================================
# 6. MODEL + LOSS
# ================================
model = MALUNet(input_channels=1, num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


# ================================
# 7. RESUME TRAINING
# ================================
if os.path.exists("malunet_best.pth"):
    model.load_state_dict(torch.load("malunet_best.pth"))
    print("Resuming from saved model...")


# ================================
# 8. TRAINING (25 EPOCHS)
# ================================
epochs = 25

for epoch in range(epochs):
    total_loss = 0
    print(f"\nStarting Epoch {epoch+1}...")

    for i, (img, mask) in enumerate(loader):
        img = img.to(device)
        mask = mask.to(device)

        pred = model(img)

        loss = loss_fn(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}")

    # 🔥 Save every epoch
    torch.save(model.state_dict(), "malunet_best.pth")
    print("Model checkpoint saved!")


# ================================
# 9. TEST + VISUALIZATION
# ================================
model.eval()

img, mask = dataset[0]
img_input = img.unsqueeze(0).to(device)

pred = model(img_input)
pred = torch.argmax(pred, dim=1).cpu().numpy()[0]

# visualize colors
color_pred = np.zeros((256,256,3), dtype=np.uint8)
color_pred[pred == 1] = [0,255,0]   # FH
color_pred[pred == 2] = [0,0,255]   # PS

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(img[0], cmap='gray')

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(mask.numpy())

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(color_pred)

plt.show()