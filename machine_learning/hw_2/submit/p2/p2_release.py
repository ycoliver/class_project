import os
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torchvision
import torch
from torchvision import transforms
np.random.seed(42)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cal_loss(pred, Y_onehot):
    return - np.mean(np.log(np.sum(np.multiply(pred, Y_onehot), axis=1)))

def load_data_img(path, classes, img_size=32):
    if os.path.exists(path+'X.npy'):
        X = np.load(path+'X.npy')
        Y = np.load(path+'Y.npy')
        return X, Y
    X, Y = [], []
    for y, cls in enumerate(classes):
        data_path = Path(path + cls)
        for p in data_path.iterdir():
            img = ImageOps.grayscale(Image.open(f"{p}"))
            img = img.resize((img_size, img_size))
            x = np.array(img).flatten()
            X.append(x)
            Y.append(y)
    X, Y = np.array(X), np.array(Y)
    np.save(path+'X.npy', X)
    np.save(path+'Y.npy', Y)
    return X, Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='weather',type=str)
    
    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name == 'weather':
        classes = os.listdir(f'./{dataset_name}/dataset')
        X_all, Y_all = load_data_img('./weather/dataset/', classes) # loading data
        # spilt
        data_num = X_all.shape[0]
        train_idx = np.random.choice(data_num, size=int(data_num*0.8), replace=False)
        test_idx = np.delete(np.arange(data_num), train_idx)
        X, Y = X_all[train_idx], Y_all[train_idx]
        X_test, Y_test = X_all[test_idx], Y_all[test_idx]
    elif dataset_name == 'coffee':
        classes = os.listdir(f'./coffee/train')
        X, Y = load_data_img('./coffee/train/',classes)
        X_test, Y_test = load_data_img('./coffee/test/',classes)
    else:
        raise 'Error dataset name'
    n, n_test = X.shape[0], X_test.shape[0]
    mu = 2e-2

    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    inlcude_bias = True
    optimizers = ['agd', 'gd']

    if inlcude_bias:
        X = np.concatenate([X, np.ones(shape=(n, 1))], axis=1)
        X_test = np.concatenate([X_test, np.ones(shape=(n_test, 1))], axis=1)

    d = X.shape[1]
    K = np.max(Y) + 1
    Y_onehot = np.eye(K)[Y]
    Y_test_onehot = np.eye(K)[Y_test]

    epochs = 1000

    results = {}

    for opt in optimizers:
        Theta = np.zeros(shape=(d, K))
        v = np.zeros(shape=(d, K))
        
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        
        for epoch in range(epochs):
            if epoch % 5 == 0:
                pred_train = softmax(np.matmul(X, Theta))
                train_loss = cal_loss(pred_train, Y_onehot)
                pred_test = softmax(np.matmul(X_test, Theta))
                test_loss = cal_loss(pred_test, Y_test_onehot)
                test_acc = np.sum(pred_test.argmax(axis=1)==Y_test) / n_test
                train_acc = np.sum(pred_train.argmax(axis=1)==Y) / n
                
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                
                print(f"[{opt.upper()}] epoch:{epoch}, train_loss:{train_loss:.5f}, test_loss:{test_loss:.5f}, test_acc:{test_acc:.4f}, train_acc:{train_acc:.4f}")
            
            pred = softmax(np.matmul(X, Theta))
            gradient = np.matmul(X.T, pred - Y_onehot) / n
            
            if opt == 'agd':
                v_new = mu * v + gradient
                Theta = Theta - v_new
                v = v_new
            elif opt == 'gd':
                Theta = Theta - mu * gradient
        
        results[opt] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
        }

    epochs_plot = np.arange(0, epochs, 5)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for opt in optimizers:
        plt.plot(epochs_plot, results[opt]['train_losses'], label=f'{opt.upper()} Train Loss')
        plt.plot(epochs_plot, results[opt]['test_losses'], label=f'{opt.upper()} Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for opt in optimizers:
        plt.plot(epochs_plot, results[opt]['train_accs'], label=f'{opt.upper()} Train Accuracy')
        plt.plot(epochs_plot, results[opt]['test_accs'], label=f'{opt.upper()} Test Accuracy', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'./{dataset_name}_results.png', dpi=300, bbox_inches='tight')
    # plt.show()