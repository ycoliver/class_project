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
# from datasets import load_dataset
import argparse
import json
from Imagenet_loader import TinyImageNet

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 残差连接
        out = F.relu(out)
        
        return out


class CNNBase(nn.Module):
    def __init__(self,
                 base_layer_num=2,
                 base_hidden_dim=1,
                 pooling_type=False,
                 is_resnet=False):
        super(CNNBase, self).__init__()
        self.base_layer_num = base_layer_num
        self.base_hidden_dim = base_hidden_dim
        self.pooling_type = 'max' if pooling_type == False else 'mean'
        self.is_resnet = is_resnet
        self.channels = [3, 6 * base_hidden_dim, 16 * base_hidden_dim]
        if base_layer_num > 2:
            for i in range(base_layer_num - 2):
                next_channel = self.channels[-1] * 2
                self.channels.append(next_channel)
        self.conv_layers = nn.ModuleList()
        if is_resnet:
            for i in range(base_layer_num):
                in_ch = self.channels[i]
                out_ch = self.channels[i + 1]
                self.conv_layers.append(ResidualBlock(in_ch, out_ch, stride=1))
        else:
            for i in range(base_layer_num):
                in_ch = self.channels[i]
                out_ch = self.channels[i + 1]
                conv_block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU()
                )
                self.conv_layers.append(conv_block)
        if self.pooling_type == 'max':
            self.pool = nn.MaxPool2d(2, 2)
        elif self.pooling_type == 'mean':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        feature_size = 64 // (2 ** base_layer_num)
        fc_input_dim = self.channels[-1] * feature_size * feature_size
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, 512 * base_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512 * base_hidden_dim, 256 * base_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256 * base_hidden_dim, 200)
        )
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.pool(x)
        x = self.fc_layers(x)
        return x

    def get_config_str(self):
        return (f"layers{self.base_layer_num}_"
                f"dim{self.base_hidden_dim}x_"
                f"{self.pooling_type}pool_"
                f"{'resnet' if self.is_resnet else 'plain'}")


def train(model, train_loader, test_loader, criterion, optimizer, 
          epochs, device, save_dir, is_l2_loss=False, l2_lambda=0.001,
          config_name="base"):

    model.train()

    train_losses = []
    test_accuracies = []
    best_acc = 0.0
    
    for epoch in range(epochs):
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
            if batch_idx % 100 == 99:
                avg_loss = running_loss / 100
                train_acc = 100 * correct / total
                print(f'Epoch: {epoch+1}/{epochs} | '
                      f'Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {avg_loss:.4f} | '
                      f'Train Acc: {train_acc:.2f}%')
                running_loss = 0.0

        test_acc = evaluate(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        print(f'\nEpoch {epoch+1} Complete - Test Accuracy: {test_acc:.2f}%\n')
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(save_dir, f'best_{config_name}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
            print(f'Best model saved with accuracy: {test_acc:.2f}%')

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'{config_name}_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
    
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


def detailed_test(model, test_loader, device, classes):
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    y_trues, y_preds = [], []
    
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
    
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'{classes[i]:12s}: {acc:5.2f}% ({class_correct[i]}/{class_total[i]})')

    overall_acc = 100 * sum(class_correct) / sum(class_total)
    print(f'\nOverall Accuracy: {overall_acc:.2f}%')
    print(classification_report(y_trues, y_preds, target_names=classes))
    
    return y_trues, y_preds, overall_acc


def plot_confusion_matrix(y_trues, y_preds, classes, save_path):
    cm = confusion_matrix(y_trues, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(history_dict, save_path):
    plt.figure(figsize=(12, 6))
    
    for config_name, accuracies in history_dict.items():
        epochs = range(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, marker='o', label=config_name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Test Accuracy Comparison Across Configurations', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


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
    
    train_loader, test_loader = tiny_loader(64, '/Users/daixunlian/workspace/class_project/deep_learning/tiny-imagenet-200')

    train_dir = '/Users/daixunlian/workspace/class_project/deep_learning/tiny-imagenet-200/train'
    classes = sorted([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    save_dir = './tiny_imagenet/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    num_epochs = 20
    learning_rate = 0.001
    
    model = CNNBase(base_layer_num=args.layer_num,
                    base_hidden_dim=args.hidden_dim,
                    pooling_type=args.mean_pooling,
                    is_resnet=args.is_resnet).to(device)
    
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
    
    # 详细测试
    y_trues, y_preds, final_acc = detailed_test(model, test_loader, device, classes)
    
    # 保存混淆矩阵
    cm_path = f'./tiny_imagenet/outputs/confusion_matrix_{config_name}.png'
    plot_confusion_matrix(y_trues, y_preds, classes, cm_path)
    print('*'*20)
    print(f"\nFinal Accuracy of The experiment: {args.exp_name} is {final_acc:.2f}%")
    with open('./tiny_imagenet/outputs/experiment_results.txt', 'a') as f:
        f.write(f"Experiment: {args.exp_name}, Final Accuracy: {final_acc:.2f}%, Training Time: {training_time:.2f} seconds\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',type=str,default='base_experiment')
    parser.add_argument('--hidden_dim',type=int,default=1)
    parser.add_argument('--layer_num',type=int,default=2)
    parser.add_argument('--mean_pooling',action='store_true',default=False)
    parser.add_argument('--is_resnet',action='store_true',default=False)
    parser.add_argument('--is_l2_loss',action='store_true',default=False)
    parser.add_argument('--use_adam',action='store_true',default=False)
    args = parser.parse_args()
    main(args)
