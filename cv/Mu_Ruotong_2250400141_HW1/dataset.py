import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from torchvision import transforms

class EG1800Dataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'Images')
        self.labels_dir = os.path.join(root_dir, 'Labels')
        
        with open(os.path.join(root_dir, f'eg1800_{split}.txt'), 'r') as f:
            self.image_names = [line.strip() for line in f]

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
        ])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.labels_dir, img_name)).convert('L')
        
        return self.img_transform(image), self.mask_transform(mask)

