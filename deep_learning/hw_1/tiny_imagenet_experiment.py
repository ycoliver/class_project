import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import time
import argparse
import json
from Imagenet_loader import TinyImageNet
from model_utils import *


def train(model, train_loader, test_loader, criterion, optimizer, 
          epochs, device, save_dir, is_l2_loss=False, l2_lambda=0.001,
          config_name="base", patience=10):
    
    model.train()
    
    train_losses = []
    test_accuracies = []
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    start_epoch = 0
    
    checkpoint_path = os.path.join(save_dir, f'best_{config_name}.pth')
    if os.path.exists(checkpoint_path):
        print(f"\n{'='*60}")
        print(f"检测到已有检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']
        best_epoch = checkpoint['epoch']
        print(f"从 Epoch {start_epoch+1} 继续训练")
        print(f"已加载最佳准确率: {best_acc:.2f}%")
        print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    print(f"训练配置: {config_name}")
    print(f"L2正则化: {'是 (λ=' + str(l2_lambda) + ')' if is_l2_loss else '否'}")
    print(f"早停机制: 启用 (patience={patience})")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            if is_l2_loss:
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += l2_lambda * l2_reg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if batch_idx % 100 == 0:
                avg_loss = running_loss / 100
                train_acc = 100 * correct / total
                print(f'Epoch: {epoch+1}/{epochs} | '
                      f'Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {avg_loss:.4f} | '
                      f'Train Acc: {train_acc:.2f}%')
                running_loss = 0.0

        test_acc = evaluate(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        print(f'\nEpoch {epoch+1} 完成 - 测试准确率: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            epochs_no_improve = 0
            
            save_path = os.path.join(save_dir, f'best_{config_name}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
            print(f'✓ 最佳模型已保存，准确率: {test_acc:.2f}%')
        else:
            epochs_no_improve += 1
            print(f'准确率未提升 ({epochs_no_improve}/{patience})')
            
            if epochs_no_improve >= patience and epoch >= 80:
                print(f'\n{"="*60}')
                print(f'早停触发！连续{patience}个epoch准确率未提升')
                print(f'最佳准确率: {best_acc:.2f}% (Epoch {best_epoch+1})')
                print(f'{"="*60}\n')
                break
        
        print()

        if epoch % 50 == 0:
            save_path = os.path.join(save_dir, f'{config_name}_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
            print(f'检查点已保存: epoch {epoch+1}')
    
    print(f'\n{"="*60}')
    print(f'训练完成！')
    print(f'最终epoch: {epoch+1}')
    print(f'最佳准确率: {best_acc:.2f}% (Epoch {best_epoch+1})')
    print(f'{"="*60}\n')
    
    return test_accuracies, best_acc


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def detailed_test(model, test_loader, device, classes, save_path=None):
    
    if save_path is not None:
        print(f"\n{'='*60}")
        print(f"加载最佳模型: {save_path}")
        print(f"{'='*60}")
        
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            saved_acc = checkpoint.get('accuracy', 'N/A')
            saved_epoch = checkpoint.get('epoch', 'N/A')
            print(f"✓ 模型加载成功")
            print(f"  - 训练epoch: {saved_epoch + 1 if isinstance(saved_epoch, int) else saved_epoch}")
            print(f"  - 保存时准确率: {saved_acc if isinstance(saved_acc, str) else f'{saved_acc:.2f}%'}")
            print(f"{'='*60}\n")
        else:
            print(f"⚠ 警告: 模型文件不存在，使用当前模型状态")
            print(f"{'='*60}\n")
    
    model.eval()
    class_correct = [0] * 200
    class_total = [0] * 200
    y_trues, y_preds = [], []
    
    print("正在进行详细测试...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_preds.extend(predicted.cpu().numpy())
            y_trues.extend(labels.cpu().numpy())
            
            c = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print("\n" + "="*50)
    print("各类别准确率 (前20个类别):")
    print("="*50)
    for i in range(len(classes)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'{classes[i]:12s}: {acc:5.2f}% ({class_correct[i]}/{class_total[i]})')
    
    overall_acc = 100 * sum(class_correct) / sum(class_total)
    print(f'\n总体准确率: {overall_acc:.2f}%')
    
    print("\n" + "="*50)
    print("分类报告 (前20个类别):")
    print("="*50)
    print(classification_report(y_trues, y_preds, target_names=classes))
    
    return y_trues, y_preds, overall_acc

def plot_confusion_matrix(y_trues, y_preds, classes, save_path):
    cm = confusion_matrix(y_trues, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.title("混淆矩阵 (Tiny ImageNet)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存到 {save_path}")

def plot_training_history(history_dict, save_path):
    plt.figure(figsize=(12, 6))
    
    for config_name, accuracies in history_dict.items():
        epochs = range(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, marker='o', label=config_name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('不同配置的测试准确率比较 (Tiny ImageNet)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练历史图已保存到 {save_path}")



def tiny_loader(batch_size, data_dir):
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
    trainset = TinyImageNet(data_dir, train=True, transform=transform_train)
    testset = TinyImageNet(data_dir, train=False, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader

def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("Loading tiny_imagenet-200 dataset...")
    
    train_loader, test_loader = tiny_loader(512, './dataset/tiny-imagenet-200')

    train_dir = './dataset/tiny-imagenet-200/train'
    classes = sorted([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    save_dir = './outputs/tiny_imagenet/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    num_epochs = 200
    learning_rate = 0.001
    
    model = CNNBase(is_large_hidden=args.is_large_hidden,
                    is_large_layer=args.is_large_layer,
                    pooling_type=args.mean_pooling,
                    is_resnet=args.is_resnet,
                    label_num=200,
                    input_size=64,
                    is_imagenet=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    if args.use_adam:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    config_name = args.exp_name
    if args.is_l2_loss:
        config_name += "_l2reg"
    
    start_time = time.time()
    accuracies, best_acc = train(
        model, train_loader, test_loader, criterion, optimizer,
        epochs=num_epochs, device=device, save_dir=save_dir,
        is_l2_loss=args.is_l2_loss,
        l2_lambda=0.001,
        config_name=config_name
    )
    training_time = time.time() - start_time
    
    y_trues, y_preds, final_acc = detailed_test(model, test_loader, device, classes, save_path=os.path.join(save_dir, f'best_{config_name}.pth'))
    
    cm_path = f'./outputs/tiny_imagenet/confusion_matrix_{config_name}.png'
    plot_confusion_matrix(y_trues, y_preds, classes, cm_path)
    print('*'*20)
    print(f"\nFinal Accuracy of The experiment: {args.exp_name} is {final_acc:.2f}%")
    with open('./outputs/tiny_imagenet/experiment_results.txt', 'a') as f:
        f.write(f"Experiment: {args.exp_name}, Final Accuracy: {final_acc:.2f}%, Training Time: {training_time:.2f} seconds\n")
    cm_path = f'./outputs/tiny_imagenet/confusion_matrix_{config_name}.png'
    plot_confusion_matrix(y_trues, y_preds, classes, cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',type=str,default='base_experiment')
    parser.add_argument('--is_large_layer',action='store_true',default=False)
    parser.add_argument('--is_large_hidden',action='store_true',default=False)
    parser.add_argument('--mean_pooling',action='store_true',default=False)
    parser.add_argument('--is_resnet',action='store_true',default=False)
    parser.add_argument('--is_l2_loss',action='store_true',default=False)
    parser.add_argument('--use_adam',action='store_true',default=False)
    args = parser.parse_args()
    main(args)