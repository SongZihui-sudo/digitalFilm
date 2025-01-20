from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import tqdm


class ImagePairDataset(Dataset):
    def __init__(self, folder1, folder2, pairs, transform=None):
        self.folder1 = folder1
        self.folder2 = folder2
        self.pairs = pairs
        self.transform = transform
        self.image_pairs = self.read_image_pairs()

    def read_image_pairs(self):
      image_pairs = []
      for image_pair in tqdm.tqdm(self.pairs):
        img1_path = os.path.join(self.folder1, image_pair[0])
        img2_path = os.path.join(self.folder2, image_pair[1])
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        image_pairs.append((img1, img2))
      return image_pairs
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
      return self.image_pairs[idx][0], self.image_pairs[idx][1]
    
def show_paired_images(dataset, batch_size=4):
  # Create DataLoader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  for img1, img2 in dataloader:
    # Convert tensor to numpy for plotting
    img1 = img1.permute(0, 2, 3, 1).numpy()  # Change to (batch_size, height, width, channels)
    img2 = img2.permute(0, 2, 3, 1).numpy()

    # Plot pairs of images
    fig, axes = plt.subplots(batch_size, 2, figsize=(8, batch_size * 4))
    for i in range(batch_size):
      axes[i, 0].imshow(img1[i])
      axes[i, 0].axis('off')  # Hide axes
      axes[i, 0].set_title(f"Image 1 - Pair {i+1}")

      axes[i, 1].imshow(img2[i])
      axes[i, 1].axis('off')  # Hide axes
      axes[i, 1].set_title(f"Image 2 - Pair {i+1}")

    plt.tight_layout()
    plt.show()
    break  # Show only one batch for simplicity