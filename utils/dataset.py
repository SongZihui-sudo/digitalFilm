import os
import cv2
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch
from PIL import Image

from utils.utils import set_random_seed


class filteredRandomCrop:
    def __init__(self, size, threshold=230, max_ghost_area=0.5, max_retries=10, seed = 42):
        self.size = (size, size) if isinstance(size, int) else size
        self.threshold = threshold
        self.max_ghost_area = max_ghost_area
        self.max_retries = max_retries
        self.random_crop = transforms.RandomCrop(size)
        self.seed = seed

    def __call__(self, img):
        set_random_seed(self.seed)
        
        crop = None
        for _ in range(self.max_retries):
            crop = self.random_crop(img)

            gray_crop = np.array(crop.convert("L"))
            white_area_ratio = np.mean(gray_crop > self.threshold)

            if white_area_ratio <= self.max_ghost_area:
                return crop

        return crop

def build_preprocessing_transforms(configs: dict, AorB: str, seed: int = 42) -> transforms.Compose:
    """
    根据配置字典构建图像预处理管道。
    """
    set_random_seed(seed)

    input_size: int = configs.input_size
    mid_reso: float = configs.mid_reso
    pre_transform_config = getattr(configs, AorB, [])
    mid_reso = round(min(mid_reso, 2) * input_size)

    preprocessing_transform: list = [
        transforms.Resize(
            mid_reso, interpolation=transforms.InterpolationMode.LANCZOS
        ),
    ]

    if "randomcrop" in pre_transform_config:
        crop_size = configs.crop_size
        preprocessing_transform.append(
            transforms.RandomCrop(crop_size)
        )

    if "filteredrandomcrop" in pre_transform_config:
        crop_size = configs.crop_size
        preprocessing_transform.append(filteredRandomCrop(
            size=crop_size, threshold=235, max_ghost_area=0.6, max_retries=15, seed=seed
        ))

    if "crop" in pre_transform_config:
        crop_size = configs.crop_size
        preprocessing_transform.append(transforms.CenterCrop(crop_size))

    if "resize" in pre_transform_config:
        crop_size = configs.crop_size
        preprocessing_transform.append(transforms.Resize((crop_size, crop_size)))

    if "horizontalflip" in pre_transform_config:
        preprocessing_transform.append(transforms.RandomHorizontalFlip(p=0.5))
    if "verticalflip" in pre_transform_config:
        preprocessing_transform.append(transforms.RandomVerticalFlip(p=0.5))

    if "color_jitter" in pre_transform_config:
        preprocessing_transform.append(
            transforms.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.2, 
                hue=0.1
            )
        )

    preprocessing_transform.append(transforms.ToTensor())

    if "Normalize" in pre_transform_config:
        preprocessing_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    # 最终组合
    return transforms.Compose(preprocessing_transform)

class filmDataset(Dataset):
    def __init__(self, opt, is_train=True):
        self.opt = opt
        self.is_train = is_train
        dataroot = opt.dataroot
        IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        if self.is_train:
            dataroot = os.path.join(dataroot, "train")
            self.dir_A = os.path.join(dataroot, 'digital')
            self.dir_B = os.path.join(dataroot, 'film')  
            self.paths_A = sorted([
                str(p) for p in Path(self.dir_A).rglob('*')
                if p.suffix.lower() in IMG_EXTENSIONS
            ])
            self.paths_B = sorted([
                str(p) for p in Path(self.dir_B).rglob('*')
                if p.suffix.lower() in IMG_EXTENSIONS
            ])
        else:
            dataroot = os.path.join(dataroot, "val")
            self.dir_A = dataroot
            self.paths_A = sorted([
                str(p) for p in Path(self.dir_A).rglob('*')
                if p.suffix.lower() in IMG_EXTENSIONS
            ])
        
        self.transforme_A: transforms.Compose = build_preprocessing_transforms(opt, "pre_transform_A", opt.seed)
        self.transforme_B: transforms.Compose = build_preprocessing_transforms(opt, "pre_transform_B", opt.seed)

        self.size = len(self.paths_A)

    def __getitem__(self, index):
        path_A: str = self.paths_A[index]
        img_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        
        img_A = Image.fromarray(img_A)

        img_B = None
        if self.is_train:
            path_B: str = self.paths_B[index % len(self.paths_B)]
            img_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
            img_B = Image.fromarray(img_B)
            
            img_A = self.transforme_A(img_A)
            img_B = self.transforme_B(img_B)
        else:
            img_A = self.transforme_A(img_A)
            
        return {
            'A': img_A, 
            'B': img_B if img_B is not None else torch.zeros_like(img_A), # DDP 通常建议返回相同结构的 Tensor
        }

    def __len__(self):
        return self.size
