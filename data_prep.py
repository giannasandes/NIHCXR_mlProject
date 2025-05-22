# data_prep.py

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# number of disease labels
NUM_LABELS = 15

def get_data_loaders(
    hf_ds_name,
    image_size=(224,224),
    batch_size=32,
    num_workers=4
):
    # ImageNet normalization
    imagenet_norm = T.Normalize(
        mean=[0.485,0.456,0.406],
        std =[0.229,0.224,0.225],
    )

    # Training augmentations
    train_tf = T.Compose([
        T.Resize(image_size, antialias=True),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        T.ColorJitter(0.2,0.2,0.2,0.02),
        T.ToTensor(),
        imagenet_norm,
        T.RandomErasing(p=0.3, scale=(0.02,0.15), ratio=(0.3,3.3), value='random'),
    ])

    # Validation / test transforms
    eval_tf = T.Compose([
        T.Resize(image_size, antialias=True),
        T.ToTensor(),
        imagenet_norm,
    ])

    # Load Hugging Face splits
    ds_train = load_dataset(hf_ds_name, 'image-classification', split='train')
    ds_test  = load_dataset(hf_ds_name, 'image-classification', split='test')

    class CXR(Dataset):
        def __init__(self, hf_ds, tf):
            self.hf = hf_ds
            self.tf = tf
        def __len__(self):
            return len(self.hf)
        def __getitem__(self, idx):
            sample = self.hf[idx]
            img = sample['image']
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            else:
                img = img.convert('RGB')
            x = self.tf(img)
            y = torch.zeros(NUM_LABELS, dtype=torch.float32)
            y[sample['labels']] = 1.0
            return x, y

    full_train = CXR(ds_train, train_tf)
    idxs = list(range(len(full_train)))
    tr_idx, va_idx = train_test_split(idxs, test_size=0.1, random_state=42)
    train_ds = Subset(full_train, tr_idx)
    val_ds   = Subset(full_train, va_idx)
    test_ds  = CXR(ds_test, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader, NUM_LABELS
