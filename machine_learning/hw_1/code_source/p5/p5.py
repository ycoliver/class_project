import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def load_mat(path, d=16):
    data = scipy.io.loadmat(path)['zip']
    size = data.shape[0]
    y = data[:, 0].astype('int')
    X = data[:, 1:].reshape(size, d, d)
    return X, y

def cal_intensity(X):
    """
    X: (n, d), input data
    return intensity: (n, 1)
    """
    n = X.shape[0]
    return np.mean(X.reshape(n, -1), 1, keepdims=True)

def cal_symmetry(X):
    """
    X: (n, d), input data
    return symmetry: (n, 1)
    """
    n, d = X.shape[:2]
    Xl = X[:, :, :int(d/2)]
    Xr = np.flip(X[:, :, int(d/2):], -1)
    abs_diff = np.abs(Xl-Xr)
    return np.mean(abs_diff.reshape(n, -1), 1, keepdims=True)

def cal_feature(data):
    intensity = cal_intensity(data)
    symmetry = cal_symmetry(data)
    feat = np.hstack([intensity, symmetry])

    return feat

def cal_feature_cls(data, label, cls_A=1, cls_B=6):
    """ calculate the intensity and symmetry feature of given classes
    Input:
        data: (n, d1, d2), the image data matrix
        label: (n, ), corresponding label
        cls_A: int, the first digit class
        cls_B: int, the second digit class
    Output:
        X: (n', 2), the intensity and symmetry feature corresponding to 
            class A and class B, where n'= cls_A# + cls_B#.
        y: (n', ), the corresponding label {-1, 1}. 1 stands for class A, 
            -1 stands for class B.
    """
    feat = cal_feature(data)
    indices = (label==cls_A) + (label==cls_B)
    X, y = feat[indices], label[indices] # filter the target number 1 and 6
    ind_A, ind_B = y==cls_A, y==cls_B
    y[ind_A] = 1
    y[ind_B] = -1

    return X, y

def plot_feature(feature, y, plot_num, ax=None, classes=np.arange(10)):
    """plot the feature of different classes
    Input:
        feature: (n, 2), the feature matrix.
        y: (n, ) corresponding label.
        plot_num: int, number of samples for each class to be plotted.
        ax: matplotlib.axes.Axes, the axes to be plotted on.
        classes: array(0-9), classes to be plotted.
    Output:
        ax: matplotlib.axes.Axes, plotted axes.
    """
    cls_features = [feature[y==i] for i in classes]

    marks = ['s', 'o', 'D', 'v', 'p', 'h', '+', 'x', '<', '>']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'cyan', 'orange', 'purple']
    if ax is None:
        _, ax = plt.subplots()
    for i, feat in zip(classes, cls_features):
        ax.scatter(*feat[:plot_num].T, marker=marks[i], color=colors[i], label=str(i))
    plt.legend(loc='upper right')
    plt.xlabel('intensity')
    plt.ylabel('symmetry')
    return ax

def cal_error(theta, X, y, thres=1e-4):
    """calculate the binary error of the model w given data (X, y)
    theta: (d+1, 1), the weight vector
    X: (n, d), the data matrix [X, y]
    y: (n, ), the corresponding label
    """
    out = X @ theta - thres
    pred = np.sign(out)
    err = np.mean(pred.squeeze()!=y)
    return err



def run_perceptron(iters, X, y, X_test, y_test, theta, threshold):
    erros = []
    test_erros = []
    for iterate in range(iters):
        ### for perceptron
        flag = False
        for i in range(num_sample):
            if y[i] * (X[i] @ theta) <= threshold:
                theta += (y[i] * X[i]).reshape(-1, 1)
                flag = True
                break
        erro = cal_error(theta, X, y, threshold)
        test_erro = cal_error(theta, X_test, y_test, threshold)
        erros.append(erro)
        test_erros.append(test_erro)
        if not flag:
            print(f'Converged at iteration {iterate}')
            break

    return theta, erros, test_erros # 返回划分好的解

def run_pocket(iters, X, y, X_test, y_test, theta, threshold):
    erros = []
    test_erros = []
    best_theta = theta.copy()
    best_err = cal_error(theta, X, y, threshold)
    for iterate in range(iters):
        flag = False
        for i in range(num_sample):
            if y[i] * (X[i] @ theta) <= threshold:
                theta += (y[i] * X[i]).reshape(-1, 1)
                current_err = cal_error(theta, X, y, threshold)
                if current_err < best_err:
                    best_err = current_err
                    best_theta = theta.copy()
                flag = True
                break
        erro = cal_error(best_theta, X, y, threshold)
        test_erro = cal_error(best_theta, X_test, y_test, threshold)
        erros.append(erro)
        test_erros.append(test_erro)
        if not flag:
            print(f'Converged at iteration {iterate}')
            break
    return best_theta, erros, test_erros # 返回划分好的解



import matplotlib.pyplot as plt

def plot_errors(errors_perceptron, test_errors_perceptron, 
                errors_pocket, test_errors_pocket, 
                filename="error_comparison.png"):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(errors_perceptron, 'b-', label='Perceptron (train)')
    plt.plot(errors_pocket, 'r-', label='Pocket (train)')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Training Error')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_errors_perceptron, 'b-', label='Perceptron (test)')
    plt.plot(test_errors_pocket, 'r-', label='Pocket (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Test Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_features_with_boundaries(feature, y, theta_perceptron, theta_pocket, 
                                 plot_num=500, filename="feature_boundary.png"):

    # 筛选1和6两类数据
    mask_1 = (y == 1)
    mask_6 = (y == -1)
    feature_sub_1 = feature[mask_1]
    feature_sub_6 = feature[mask_6]
    y_sub_1 = y[mask_1]
    y_sub_6 = y[mask_6]
    
    # 创建绘图
    plt.figure(figsize=(10, 8))
    
    # 绘制1类数据
    class1 = feature_sub_1[y_sub_1 == 1]
    plt.scatter(class1[:plot_num, 0], class1[:plot_num, 1], 
                marker='s', color='r', label='1')
    
    # 绘制6类数据
    class6 = feature_sub_6[y_sub_6 == -1]
    plt.scatter(class6[:plot_num, 0], class6[:plot_num, 1], 
                marker='o', color='b', label='6')
    
    # 获取当前坐标范围
    x_min, x_max = plt.xlim()
    
    
    # 绘制感知器分类边界
    x_vals = np.array([x_min, x_max])
    # 边界方程: theta0 x0 + theta1 x1 = 0
    y_perceptron = - theta_perceptron[0] * x_vals / theta_perceptron[1]
    plt.plot(x_vals, y_perceptron, 'g-', linewidth=2, label='Perceptron Boundary')
    
    # 绘制口袋算法分类边界
    y_pocket = -theta_pocket[0] *x_vals/theta_pocket[1]
    plt.plot(x_vals, y_pocket, 'm--', linewidth=2, label='Pocket Boundary')
    
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.title('Classification Boundaries (1 vs 6)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # prepare data
    train_data, train_label = load_mat('train_data.mat') # train_data: (7291, 16, 16), train_label: (7291, )
    test_data, test_label = load_mat('test_data.mat') # test_data: (2007, 16, 16), train_label: (2007, )

    cls_A, cls_B = 1, 6
    X, y, = cal_feature_cls(train_data, train_label, cls_A=cls_A, cls_B=cls_B)
    X_test, y_test = cal_feature_cls(test_data, test_label, cls_A=cls_A, cls_B=cls_B)

    train_feat = cal_feature(train_data)
    plot_feature(train_feat, train_label, 10)
    # plt.show()

    # train
    iters = 2000 # 判断是否收敛
    d = 2
    num_sample = X.shape[0]
    threshold = 1e-4
    theta = np.zeros((d, 1)) # 斜率和截距

    theta_perceptron, erros_perceptron, test_erros_perceptron = run_perceptron(iters, X, y, X_test, y_test, theta, threshold)
    theta_pocket, erros_pocket, test_erros_pocket = run_pocket(iters, X, y, X_test, y_test, theta, threshold)
    plot_errors(erros_perceptron, test_erros_perceptron, erros_pocket, test_erros_pocket, filename="error_comparison.png")

    plot_features_with_boundaries(X, y, theta_perceptron, theta_pocket, plot_num=500, filename="feature_boundary.png")