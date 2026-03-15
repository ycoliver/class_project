import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import EG1800Dataset
from model import UNet

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_loss = total_loss / len(dataloader)
    pixel_acc = correct_pixels / total_pixels
    
    return avg_loss, pixel_acc

def main():
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    IMG_SIZE = 224
    NUM_CLASSES = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集
    train_dataset = EG1800Dataset(
        root_dir='./EG1800',
        split='train',
        img_size=IMG_SIZE
    )
    
    test_dataset = EG1800Dataset(
        root_dir='./EG1800',
        split='test',
        img_size=IMG_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f'Train dataset: {len(train_dataset)} images')
    print(f'Test dataset: {len(test_dataset)} images')
    
    # model
    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)

    print(model)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    # loss function, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Pixel Acc: {val_acc:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet.pth')
            print('save model')
    
    print('\nTraining completed!')

if __name__ == '__main__':
    main()