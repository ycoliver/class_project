import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import EG1800Dataset
from model import UNet

def predict_and_visualize(model, dataset, device, num_samples=4):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]
            
            # 预测
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu()
            
            # 反归一化显示图像
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)
            
            # 显示
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Label Mask')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

model = UNet(n_channels=3, n_classes=2)
model.load_state_dict(torch.load('best_unet.pth'))
model.to('cuda')

test_dataset = EG1800Dataset('./EG1800', 'test', img_size=224)
predict_and_visualize(model, test_dataset, 'cuda', num_samples=4)