import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import tqdm
from sklearn.model_selection import train_test_split

import tools.mydataset as mydataset
import mynet
import tools.cal_psnr as cal_psnr

# Define paths to the folders
digital_dir = "/root/autodl-tmp/data/胶片-数码/数码"
film_dir = "/root/autodl-tmp/data/胶片-数码/胶片"

image_width = 320
image_hight = 200

batch_size = 8

# TRAIN
num_epochs = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print("DigitalFile")
    # Get the list of files in both folders
    digital = sorted(os.listdir(digital_dir))
    film = sorted(os.listdir(film_dir))

    # Ensure the number of files match
    if len(digital) != len(film):
        raise ValueError("The two folders must have the same number of images.")

    # Create pairs of images (file1, file2)
    pairs = list(zip(digital, film))

    # Split into training and testing sets
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    # Define transformations (if needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_hight, image_width))
    ])
    
    # Create datasets
    print("加载数据集中")
    train_dataset = mydataset.ImagePairDataset(digital_dir, film_dir, train_pairs, transform=transform)
    test_dataset = mydataset.ImagePairDataset(digital_dir, film_dir, test_pairs, transform=transform)
    print("加载数据集完成")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8)
    
    print("当前设备：")
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    
    model = mynet.ImageToImageCNN()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)  # make parallel
        cudnn.benchmark = True
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # LOSS FUNCTION

    print("训练中")
    model.train()
    with tqdm.tqdm(total=num_epochs, desc="进度条") as pbar:
        for epoch in range(num_epochs):
            for i, (img1, img2) in enumerate(train_loader):
                img1, img2 = img1.to(device), img2.to(device)

                optimizer.zero_grad()
                output = model(img2)

                loss = criterion(output, img1)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

    pbar.update(1)

    print("训练完成")
    
    torch.save(model.state_dict(), 'model/digital_kodak_200_film_generate.pth')
    
    print("测试")
    
    model.eval()
    
    avg_psnr = cal_psnr.evaluate_psnr(test_loader, device, model)
    print(f'Average PSNR for the dataset: {avg_psnr:.2f} dB')
            
    