#!/usr/bin/env python3
"""
测试TinyImageNet数据加载器
"""
import torch
from Imagenet_loader import TinyImageNet
from torch.utils.data import DataLoader
from torchvision import transforms

def test_tiny_imagenet_loader():
    print("测试TinyImageNet数据加载器...")
    
    # 设置数据变换
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_test = transforms.Compose([
        transforms.Resize(32), 
        transforms.ToTensor(), 
        normalize
    ])
    
    # 测试训练集
    print("\n1. 测试训练集...")
    try:
        trainset = TinyImageNet('./dataset/tiny-imagenet-200', train=True, transform=transform_test)
        print(f"训练集大小: {len(trainset)}")
        
        # 测试获取一个样本
        sample, label = trainset[0]
        print(f"训练集样本形状: {sample.shape}, 标签: {label}")
        print("✓ 训练集加载成功")
    except Exception as e:
        print(f"✗ 训练集加载失败: {e}")
        return False
    
    # 测试验证集
    print("\n2. 测试验证集...")
    try:
        testset = TinyImageNet('./dataset/tiny-imagenet-200', train=False, transform=transform_test)
        print(f"验证集大小: {len(testset)}")
        
        # 测试获取一个样本
        sample, label = testset[0]
        print(f"验证集样本形状: {sample.shape}, 标签: {label}")
        print("✓ 验证集加载成功")
    except Exception as e:
        print(f"✗ 验证集加载失败: {e}")
        return False
    
    # 测试DataLoader
    print("\n3. 测试DataLoader...")
    try:
        train_loader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)
        test_loader = DataLoader(testset, batch_size=4, shuffle=False, drop_last=False)
        
        print(f"训练集批次数量: {len(train_loader)}")
        print(f"验证集批次数量: {len(test_loader)}")
        
        # 测试获取一个批次
        for inputs, labels in train_loader:
            print(f"训练批次形状: {inputs.shape}, 标签形状: {labels.shape}")
            break
            
        for inputs, labels in test_loader:
            print(f"验证批次形状: {inputs.shape}, 标签形状: {labels.shape}")
            break
            
        print("✓ DataLoader测试成功")
    except Exception as e:
        print(f"✗ DataLoader测试失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！数据加载器修复成功。")
    return True

if __name__ == "__main__":
    test_tiny_imagenet_loader()
